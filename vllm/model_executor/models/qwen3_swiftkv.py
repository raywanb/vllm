# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.attention import AttentionMetadata, AttentionType, Attention
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3MLP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              extract_layer_index,
                                              is_pp_missing_parameter,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3SwiftKVConfig
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)

logger = init_logger(__name__)


# SwiftKVMetadata not needed - using standard vLLM attention!


class Qwen3SwiftKVAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=qkv_bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )
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
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {},
        )
        self.attn_type = attn_type
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query, _ = self.q_proj_swiftkv(hidden_states)
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_dim)
        query = self.q_norm(query)
        dummy_k = torch.empty_like(query[:, :self.num_kv_heads, :])
        query, _ = self.rotary_emb(positions, query, dummy_k)

        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_dim)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_dim)

        # Use vLLM's optimized attention layer (just like standard Qwen3!)
        # This handles all the complex KV caching, metadata, and optimization automatically
        query = query.view(-1, self.num_heads * self.head_dim)
        if key is not None:
            key = key.view(-1, self.num_kv_heads * self.head_dim)
        if value is not None:
            value = value.view(-1, self.num_kv_heads * self.head_dim)
            
        attn_output = self.attn(query, key, value)

        attn_output = attn_output.view(num_tokens, hidden_size)
        output, _ = self.o_proj(attn_output)
        return output



class Qwen3SwiftKVDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3SwiftKVConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 1_000_000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        qkv_bias = getattr(config, "attention_bias", False)
        head_dim = getattr(config, "head_dim", None)

        attn_type = "decoder"
        if not getattr(config, "is_causal", True):
            attn_type = "encoder"

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
            dual_chunk_attention_config=dual_chunk_attention_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
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
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            key=k_states,
            value=v_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _padded_size(size: int) -> int:
    mult = (1 << (size - 1).bit_length()) // 4
    if mult < 1:
        return size
    return (size + mult - 1) // mult * mult


class Qwen3SwiftKVModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if not vllm_config.scheduler_config.chunked_prefill_enabled:
            raise ValueError("SwiftKV requires chunked prefill to be enabled")

        super().__init__()

        config: Qwen3SwiftKVConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.key_value_group_size = getattr(config, "key_value_group_size", 1)
        self.kv_cache_dtype = (cache_config.cache_dtype
                               if cache_config is not None else "auto")

        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )

        self.layers = torch.nn.ModuleList([
            Qwen3DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            ) if idx < config.num_key_value_layers else Qwen3SwiftKVDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_swiftkv = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # CUDA graphs handled by standard vLLM attention layers

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

# Metadata conversion not needed - using standard vLLM attention!

    def _run_swiftkv_layers(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        shared_k: torch.Tensor,
        shared_v: torch.Tensor,
    ) -> torch.Tensor:
        """Run SwiftKV layers using shared K,V states for maximum efficiency."""
        
        # Process all SwiftKV layers with the same shared K,V
        for layer_idx in range(self.config.num_key_value_layers,
                               self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            
            # All layers use the SAME shared K,V (this is the SwiftKV optimization!)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                shared_k,  # Same K for all SwiftKV layers
                shared_v,  # Same V for all SwiftKV layers  
                residual,
            )
            
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if self.config.num_key_value_layers == self.config.num_hidden_layers:
            raise ValueError("Qwen3SwiftKVConfig requires fewer KV layers than"
                             " total layers for SwiftKV to be meaningful.")

        # No need for complex metadata handling - vLLM's attention layer handles everything!
    
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        # Process standard layers (with regular attention)
        for layer_idx in range(self.config.num_key_value_layers):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        # TRUE SwiftKV: Generate shared K,V states ONCE for ALL SwiftKV layers
        swiftkv_hidden_states = self.norm_swiftkv(hidden_states + residual)
        
        # Use the first SwiftKV layer to compute shared K,V (only once!)
        first_swiftkv_layer = self.layers[self.config.num_key_value_layers]
        attn = first_swiftkv_layer.self_attn
        kv, _ = attn.kv_proj_swiftkv(swiftkv_hidden_states)
        shared_k, shared_v = kv.split(attn.kv_size, dim=-1)
        
        # Ensure contiguous memory layout for optimal performance
        shared_k = shared_k.contiguous().view(-1, attn.num_kv_heads, attn.head_dim)
        shared_k = attn.k_norm(shared_k)
        
        # Apply RoPE to shared K (reuse dummy tensor for efficiency)
        dummy_q = torch.empty(shared_k.size(0), attn.num_heads, attn.head_dim,
                             device=shared_k.device, dtype=shared_k.dtype)
        _, shared_k = attn.rotary_emb(positions, dummy_q, shared_k)
        shared_v = shared_v.contiguous().view(-1, attn.num_kv_heads, attn.head_dim)
        
        # Process SwiftKV layers with shared K,V states (vLLM handles KV caching automatically!)
        hidden_states = self._run_swiftkv_layers(
            positions,
            hidden_states,
            residual,
            shared_k,
            shared_v,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        for layer_idx in range(self.config.num_key_value_layers):
            prefix = f".{layer_idx}.self_attn"
            stacked_params_mapping.extend([
                (f"{prefix}.qkv_proj", f"{prefix}.q_proj", "q"),
                (f"{prefix}.qkv_proj", f"{prefix}.k_proj", "k"),
                (f"{prefix}.qkv_proj", f"{prefix}.v_proj", "v"),
            ])
        for layer_idx in range(self.config.num_key_value_layers,
                               self.config.num_hidden_layers):
            prefix = f".{layer_idx}.self_attn"
            stacked_params_mapping.extend([
                (f"{prefix}.kv_proj_swiftkv", f"{prefix}.k_proj_swiftkv", "k"),
                (f"{prefix}.kv_proj_swiftkv", f"{prefix}.v_proj_swiftkv", "v"),
            ])
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            orig_name = name
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning("Skip loading %s", orig_name)
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning("Skip loading %s", orig_name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        pass


class Qwen3SwiftKVForCausalLM(nn.Module):
    packed_modules_mapping = {
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    column_parallel_weights_modules = [
        ".q_proj_swiftkv.",
        ".down_proj.",
        ".o_proj.",
    ]

    bitsandbytes_stacked_params_mapping = {
        "k_proj_swiftkv": ("kv_proj_swiftkv", 1),
        "v_proj_swiftkv": ("kv_proj_swiftkv", 2),
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.model = Qwen3SwiftKVModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE
                          if not lora_config else
                          lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids,
                          positions,
                          intermediate_tensors,
                          inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)
