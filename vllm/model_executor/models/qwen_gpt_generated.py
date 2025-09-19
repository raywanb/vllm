# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The vLLM project
#
# Qwen3 + SwiftKV model executor (vLLM v1-style)
#
# Key points:
# - Uses standard vLLM Attention to manage paged KV cache; no direct kernel/cache ops.
# - Forward signatures match current Qwen3: layer.forward(positions, hidden_states, residual)
# - Tail layers (after num_key_value_layers) use a lightweight path:
#     * Precompute K/V from a normalized hidden once per step (SwiftKV)
#     * In attention, only project Q and call self.attn(q, k, v)
# - Preserves Qwen3 specifics: q/k RMSNorm, RoPE (incl. dual-chunk), bias flags.

from collections.abc import Iterable
from typing import Any, Optional, Union, Tuple, List

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    DEFAULT_VOCAB_PADDING_SIZE,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3DecoderLayer  # reuse as head-layer implementation
from .utils import (AutoWeightsLoader, PPMissingLayer, maybe_prefix)

logger = init_logger(__name__)


# =========================
# SwiftKV Tail Attention
# =========================

class Qwen3SwiftKVAttention(nn.Module):
    """
    Tail-layer attention for Qwen3 that:
      - Projects Q only (q_proj_swiftkv).
      - Uses precomputed K/V injected by the model (set each step).
      - Applies q/k per-head RMSNorm and RoPE (same as Qwen3Attention).
      - Calls vLLM Attention(q, k, v), which manages the paged KV cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: Optional[int],
        rms_norm_eps: float,
        qkv_bias: bool,
        rope_theta: float,
        rope_scaling: Optional[dict],
        quant_config: Optional[QuantizationConfig],
        attn_type: str,
        dual_chunk_attention_config: Optional[dict],
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.hidden_size = hidden_size
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Q-only projection (tail path)
        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=qkv_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )

        # K/V projection lives here for weight ownership; model calls it.
        self.kv_proj_swiftkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj_swiftkv",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Qwen3: per-head RMSNorm on q and k
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE (supports dual-chunk if present in config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # vLLM Attention handles paged cache and kernel selection.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **({"layer_idx": _extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config}
               if dual_chunk_attention_config else {}),
        )

        # Placeholder for model-injected K/V (set every step)
        self._kv_for_step: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def set_swiftkv_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Called by the model before executing the layer; shapes:
           k, v: [T, num_kv_heads, head_dim] (k already k_norm + RoPE).
        """
        self._kv_for_step = (k, v)

    def clear_swiftkv_kv(self) -> None:
        self._kv_for_step = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self._kv_for_step is not None, \
            "SwiftKV k/v not set for this layer step"

        # ---- Q path: project, q_norm, RoPE ----
        q, _ = self.q_proj_swiftkv(hidden_states)  # [T, H*D]
        T = q.shape[0]
        q = q.view(T, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        # Rotate q (we already rotated k during KV precompute)
        # Create a dummy k placeholder for API compatibility
        dummy_k = torch.empty_like(q[:, :self.num_kv_heads, :])
        q, _ = self.rotary_emb(positions, q, dummy_k)

        # ---- Use precomputed K/V ----
        k, v = self._kv_for_step
        # vLLM Attention handles cache + matmul
        attn_out = self.attn(q, k, v)  # [T, H, D]
        attn_out = attn_out.reshape(T, self.num_heads * self.head_dim)
        out, _ = self.o_proj(attn_out)

        # One-shot usage
        self.clear_swiftkv_kv()
        return out


# =========================
# SwiftKV Tail DecoderLayer
# =========================

class Qwen3SwiftKVDecoderLayer(nn.Module):
    """Tail decoder layer that consumes precomputed K/V via its attention."""

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # Qwen3 specifics
        rope_theta = getattr(config, "rope_theta", 1_000_000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        qkv_bias = getattr(config, "attention_bias", False)
        head_dim = getattr(config, "head_dim", None)

        # Tail attention
        attn_type = AttentionType.DECODER if getattr(config, "is_causal", True)\
            else AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3SwiftKVAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            quant_config=quant_config,
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quantization_or_none(quant_config),
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Standard Qwen3 pre/post norm pattern
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states,
                                                           residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# =========================
# SwiftKV Model Core
# =========================

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is (seq_len,) for text; qwen2-vl uses (3, seq_len)
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3SwiftKVModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen3Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        # Embedding (+optional LoRA vocab extension)
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quantization_or_none(quant_config),
        )

        # Build layers: head uses standard Qwen3DecoderLayer; tail uses SwiftKV
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                config=config,
                cache_config=None,                 # Attention manages cache
                quant_config=quantization_or_none(quant_config),
                prefix=f"{prefix}.layers.{idx}",
            ) if idx < config.num_key_value_layers else Qwen3SwiftKVDecoderLayer(
                config=config,
                quant_config=quantization_or_none(quant_config),
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_swiftkv = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Exposed for wrapper
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(self, batch_size: int
                                        ) -> IntermediateTensors:
        # Mirror Qwen3 implementation if you use aux tensors; minimal here:
        return IntermediateTensors([])

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Embed
        hidden_states = (inputs_embeds
                         if inputs_embeds is not None
                         else self.get_input_embeddings(input_ids))
        residual = None
        cfg = self.config

        # 1) Head: standard layers, which fill KV cache via Attention internally
        for idx in range(cfg.num_key_value_layers):
            hidden_states, residual = self.layers[idx](
                positions, hidden_states, residual)

        # 2) Precompute per-tail-layer K/V once (SwiftKV)
        # Normalize once to feed kv-proj for all tail layers
        swiftkv_hidden = self.norm_swiftkv(hidden_states + residual)

        for idx in range(cfg.num_key_value_layers, cfg.num_hidden_layers):
            layer: Qwen3SwiftKVDecoderLayer = self.layers[idx]
            attn: Qwen3SwiftKVAttention = layer.self_attn

            # kv projection on normalized hidden
            kv, _ = attn.kv_proj_swiftkv(swiftkv_hidden)  # [T, 2*kv_size]
            k, v = kv.split(attn.kv_size, dim=-1)

            # Shape to [T, num_kv_heads, head_dim]
            T = k.shape[0]
            k = k.view(T, attn.num_kv_heads, attn.head_dim)
            v = v.view(T, attn.num_kv_heads, attn.head_dim)

            # k_norm then RoPE on k (q gets RoPE later in attention)
            k = attn.k_norm(k)
            # pass dummy q to satisfy API; only k is used from the pair
            dummy_q = torch.empty(
                (T, attn.num_heads, attn.head_dim),
                device=k.device, dtype=k.dtype)
            _, k = attn.rotary_emb(positions, dummy_q, k)

            # Inject for this step
            attn.set_swiftkv_kv(k, v)

        # 3) Tail: each layer pulls its precomputed k,v and runs
        for idx in range(cfg.num_key_value_layers, cfg.num_hidden_layers):
            hidden_states, residual = self.layers[idx](
                positions, hidden_states, residual)

        # 4) Final RMSNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# =========================
# Causal LM Wrapper
# =========================

class Qwen3SwiftKVForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        # for loader/quantization stack mapping
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


    # column-parallel in TP
    column_parallel_weights_modules = [
        ".q_proj_swiftkv.",
        ".down_proj.",
        ".o_proj.",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen3Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.model = Qwen3SwiftKVModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # PP-friendly lm_head (same pattern as Qwen3ForCausalLM)
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quantization_or_none(quant_config),
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()


        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Mirror Qwen3ForCausalLM
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    # optional helper used by Eagle3 tools
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # Tailored selection for auxiliary heads, if needed
        pass

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


# ============ helpers ============

def _extract_layer_index(prefix: str) -> int:
    # expecting prefix like "...layers.<idx>...."
    try:
        parts = prefix.split(".")
        i = parts.index("layers")
        return int(parts[i + 1])
    except Exception:
        return -1


def quantization_or_none(q: Optional[QuantizationConfig]
                         ) -> Optional[QuantizationConfig]:
    return q if q is not None else None
