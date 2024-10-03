from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

try:
    from flashinfer.cascade import MultiLevelCascadeAttentionWrapper
    from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

    import vllm.attention.backends.flash_attn  # noqa
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchDecodeWithPagedKVCacheWrapper = None
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper = None
    BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.utils import (async_tensor_h2d, get_kv_cache_torch_dtype,
                        make_tensor_with_pad)

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


class FlashInferBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "flashinfer"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashInferState"]:
        return FlashInferState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128, 256]

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


class FlashInferState(AttentionState):

    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False
        self._workspace_buffer = None
        self._wrapper = None
        self._normal_wrapper = None

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def _get_wrapper(self):
        if self._wrapper is None:
            self._wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), "NHD")
        return self._wrapper

    def _get_normal_wrapper(self):
        if self._normal_wrapper is None:
            self._normal_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), "NHD"
            )
        return self._normal_wrapper
    
    @contextmanager
    def graph_capture(self, max_batch_size: int):
        raise NotImplementedError(
            "current implementation does not support CUDA Graph")
        # self._is_graph_capturing = True
        # self._graph_decode_wrapper = None
        # self._graph_slot_mapping = torch.full((max_batch_size, ),
        #                                       PAD_SLOT_ID,
        #                                       dtype=torch.long,
        #                                       device=self.runner.device)
        # self._graph_seq_lens = torch.ones(max_batch_size,
        #                                   dtype=torch.int32,
        #                                   device=self.runner.device)
        # self._graph_block_tables = torch.from_numpy(
        #     self.runner.graph_block_tables).to(device=self.runner.device)
        # self._graph_decode_workspace_buffer = self._get_workspace_buffer()
        # self._graph_indices_buffer = torch.empty(
        #     max_batch_size * self.runner.cache_config.num_gpu_blocks,
        #     dtype=torch.int32,
        #     device=self.runner.device)
        # self._graph_indptr_buffer = torch.empty(max_batch_size + 1,
        #                                         dtype=torch.int32,
        #                                         device=self.runner.device)
        # self._graph_last_page_len_buffer = torch.empty(
        #     max_batch_size, dtype=torch.int32, device=self.runner.device)
        # yield
        # self._is_graph_capturing = False
        # del self._graph_slot_mapping
        # del self._graph_seq_lens
        # del self._graph_block_tables
        # del self._graph_decode_workspace_buffer
        # del self._graph_indices_buffer
        # del self._graph_indptr_buffer
        # del self._graph_last_page_len_buffer
        # del self._graph_decode_wrapper

    def graph_clone(self, batch_size: int):
        raise NotImplementedError(
            "current implementation does not support CUDA Graph")
        # assert self._is_graph_capturing
        # state = self.__class__(self.runner)
        # state._workspace_buffer = self._graph_decode_workspace_buffer
        # state._decode_wrapper = self._graph_decode_wrapper
        # state._prefill_wrapper = self._get_prefill_wrapper()
        # return state

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        raise NotImplementedError(
            "current implementation does not support CUDA Graph")
        # _indptr_buffer = self._graph_indptr_buffer[:batch_size + 1]
        # _last_page_len_buffer = self._graph_last_page_len_buffer[:batch_size]

        # num_qo_heads = (self.runner.model_config.get_num_attention_heads(
        #     self.runner.parallel_config))
        # num_kv_heads = self.runner.model_config.get_num_kv_heads(
        #     self.runner.parallel_config)
        # use_tensor_cores = num_qo_heads // num_kv_heads > 4
        # self._graph_decode_wrapper = \
        #     CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        #     self._graph_decode_workspace_buffer, _indptr_buffer,
        #     self._graph_indices_buffer, _last_page_len_buffer, "NHD",
        #     use_tensor_cores)
        # if self.runner.kv_cache_dtype.startswith("fp8"):
        #     kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
        #         self.runner.kv_cache_dtype)
        # else:
        #     kv_cache_dtype = get_kv_cache_torch_dtype(
        #         self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        # paged_kv_indptr_tensor_host = torch.arange(0,
        #                                            batch_size + 1,
        #                                            dtype=torch.int32)
        # paged_kv_indices_tensor_host = torch.arange(0,
        #                                             batch_size,
        #                                             dtype=torch.int32)
        # paged_kv_last_page_len_tensor_host = torch.full(
        #     (batch_size,), self.runner.block_size, dtype=torch.int32
        # )
        # query_start_loc_host = torch.arange(0,
        #                                     batch_size + 1,
        #                                     dtype=torch.int32)

        # attn_metadata = self.runner.attn_backend.make_metadata(
        #     num_prefills=0,
        #     slot_mapping=self._graph_slot_mapping[:batch_size],
        #     num_prefill_tokens=0,
        #     num_decode_tokens=batch_size,
        #     max_prefill_seq_len=0,
        #     block_tables=self._graph_block_tables,
        #     paged_kv_indptr=paged_kv_indptr_tensor_host,
        #     paged_kv_indices=paged_kv_indices_tensor_host,
        #     paged_kv_last_page_len=paged_kv_last_page_len_tensor_host,
        #     num_qo_heads=num_qo_heads,
        #     num_kv_heads=num_kv_heads,
        #     head_dim=self.runner.model_config.get_head_size(),
        #     page_size=self.runner.block_size,
        #     seq_start_loc=None,
        #     query_start_loc=query_start_loc_host,
        #     device=self.runner.device,
        #     data_type=kv_cache_dtype,
        #     q_data_type=self.runner.model_config.dtype,
        #     use_cuda_graph=True,
        #     decode_wrapper=self._graph_decode_wrapper,
        #     prefill_wrapper=None)
        # attn_metadata.begin_forward()
        # return attn_metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        return {
            "slot_mapping": attn_metadata.slot_mapping,
        }

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        return

    def begin_forward(self, model_input, model):
        assert not self._is_graph_capturing
        state = self

        try:
            scale = getattr(model.model.layers[0].self_attn.attn.impl, "scale",
                            None)
        except AttributeError as e:
            raise AttributeError("Failed to retrieve 'scale'. \
                    Check if 'self_attn.attn.impl' contains 'scale'.") from e

        try:
            logits_soft_cap = getattr(
                model.model.layers[0].self_attn.attn.impl, "logits_soft_cap",
                None)
        except AttributeError as e:
            raise AttributeError("Failed to retrieve 'logits_soft_cap'. \
                    Check if 'self_attn.attn.impl' contains 'logits_soft_cap'."
                                 ) from e

        if model_input.attn_metadata.use_cuda_graph:
            raise NotImplementedError(
                "Current implementation does not support CUDA Graph")
            # batch_size = model_input.input_tokens.shape[0]
            # state = (self.runner.graph_runners[model_input.virtual_engine]
            #          [batch_size].attn_state)

        model_input.attn_metadata.wrapper = state._get_wrapper()
        model_input.attn_metadata.normal_wrapper = state._get_normal_wrapper()
        model_input.attn_metadata.begin_forward(scale, logits_soft_cap)


@dataclass
class FlashInferMetadata(AttentionMetadata):
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int

    use_cuda_graph: bool = True

    wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None
    normal_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None
    # Metadata for wrapper
    seq_start_loc: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    second_level_query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # used for GPU in-place advance_step
    seq_lens_tensor: Optional[torch.Tensor] = None
    block_table_bound: Optional[torch.Tensor] = None

    # Refer to: https://docs.flashinfer.ai/tutorials/kv_layout.html
    # and: https://docs.flashinfer.ai/api/python/cascade.html
    # Store shared prefix blocks of requests
    paged_kv_indices: Optional[torch.Tensor] = None
    # Index pointers to the start of each shared block of KV-Cache
    paged_kv_indptr: Optional[torch.Tensor] = None
    # paged_kv_last_page_len is the length of the last page of the shared KVs
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # Store the concatenated page indices of all requests for the second layer
    second_layer_kv_indices: Optional[torch.Tensor] = None
    # Index pointers to the start of each request's page indices
    # in the second_layer_kv_indices
    second_layer_kv_indptr: Optional[torch.Tensor] = None
    # The length of the last page of each request in the second layer
    second_layer_kv_last_page_len: Optional[torch.Tensor] = None

    normal_paged_kv_indices: Optional[torch.Tensor] = None
    normal_paged_kv_indptr: Optional[torch.Tensor] = None
    normal_paged_kv_last_page_len: Optional[torch.Tensor] = None
    normal_second_layer_kv_indices: Optional[torch.Tensor] = None
    normal_second_layer_kv_indptr: Optional[torch.Tensor] = None
    normal_second_layer_kv_last_page_len: Optional[torch.Tensor] = None

    normal_query_start_loc: Optional[torch.Tensor] = None
    normal_second_level_query_start_loc: Optional[torch.Tensor] = None

    # The number of query/output heads
    num_qo_heads: Optional[int] = None
    # The number of key/value heads
    num_kv_heads: Optional[int] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    device: torch.device = torch.device("cuda")
    is_profile_run: bool = False

    def __post_init__(self):
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        supported_head_sizes = FlashInferBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.")

    def begin_forward(self, scale: Optional[float],
                      logits_soft_cap: Optional[float]):
        """
        Prepares wrapper inputs
        """

        if self.num_prefill_tokens > 0:
            if self.paged_kv_indices is None:
                return

            assert self.wrapper is not None
            assert self.query_start_loc is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            assert self.second_layer_kv_indices is not None
            assert self.second_layer_kv_indptr is not None
            assert self.second_layer_kv_last_page_len is not None
            assert self.block_table_bound is not None
            assert self.seq_lens_tensor is not None
            batch_size = self.query_start_loc.shape[0] - 1
            assert batch_size >= 0

            # We will use flash attention for profiling to
            # determine the number of blocks. Therefore,
            # we don't need to prepare the input for flashinfer for profile run.

            if not self.is_profile_run:
                self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
                self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                    self.device)
                self.block_table_bound = self.block_table_bound.to(self.device)
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)
                self.paged_kv_indices = self.paged_kv_indices.to(self.device)
                self.second_layer_kv_indices = self.second_layer_kv_indices.to(
                    self.device)
                self.second_layer_kv_indptr = self.second_layer_kv_indptr.to(
                    self.device)
                self.second_layer_kv_last_page_len = self.second_layer_kv_last_page_len.to(  # noqa: E501
                    self.device)

                self.normal_paged_kv_indices = self.normal_paged_kv_indices.to(self.device)
                self.normal_paged_kv_indptr = self.normal_paged_kv_indptr.to(self.device)
                self.normal_paged_kv_last_page_len = self.normal_paged_kv_last_page_len.to(self.device)
                self.normal_second_layer_kv_indices = self.normal_second_layer_kv_indices.to(self.device)
                self.normal_second_layer_kv_indptr = self.normal_second_layer_kv_indptr.to(self.device)
                self.normal_second_layer_kv_last_page_len = self.normal_second_layer_kv_last_page_len.to(self.device)

                self.wrapper.plan(
                    [self.query_start_loc, self.second_level_query_start_loc],
                    [self.paged_kv_indptr, self.second_layer_kv_indptr],
                    [self.paged_kv_indices, self.second_layer_kv_indices], [
                        self.paged_kv_last_page_len,
                        self.second_layer_kv_last_page_len
                    ],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    causal=True,
                    sm_scale=scale,
                    logits_soft_cap=logits_soft_cap)

                self.normal_wrapper.plan(
                    [self.normal_query_start_loc, self.normal_second_level_query_start_loc],
                    [self.normal_paged_kv_indptr, self.normal_second_layer_kv_indptr], 
                    [self.normal_paged_kv_indices, self.normal_second_layer_kv_indices], 
                    [self.normal_paged_kv_last_page_len, self.normal_second_layer_kv_last_page_len], 
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    causal=True,
                    sm_scale=scale,
                    logits_soft_cap=logits_soft_cap
                )

        else:
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            assert self.second_layer_kv_indices is not None
            assert self.second_layer_kv_indptr is not None
            assert self.second_layer_kv_last_page_len is not None

            self.normal_paged_kv_indices = self.normal_paged_kv_indices.to(self.device)
            self.normal_paged_kv_indptr = self.normal_paged_kv_indptr.to(self.device)
            self.normal_paged_kv_last_page_len = self.normal_paged_kv_last_page_len.to(self.device)
            self.normal_second_layer_kv_indices = self.normal_second_layer_kv_indices.to(self.device)
            self.normal_second_layer_kv_indptr = self.normal_second_layer_kv_indptr.to(self.device)
            self.normal_second_layer_kv_last_page_len = self.normal_second_layer_kv_last_page_len.to(self.device)

            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
            self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            self.second_layer_kv_indices = self.second_layer_kv_indices.to(
                self.device)
            self.second_layer_kv_indptr = self.second_layer_kv_indptr.to(
                self.device)
            self.second_layer_kv_last_page_len = self.second_layer_kv_last_page_len.to(  # noqa: E501
                self.device)

            # handle model warmup path
            if self.block_table_bound is not None:
                self.block_table_bound = self.block_table_bound.to(self.device)
            if self.seq_lens_tensor is not None:
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)

            assert self.wrapper is not None

            self.wrapper.plan(
                [self.query_start_loc, self.second_level_query_start_loc],
                [self.paged_kv_indptr, self.second_layer_kv_indptr],
                [self.paged_kv_indices, self.second_layer_kv_indices], [
                    self.paged_kv_last_page_len,
                    self.second_layer_kv_last_page_len
                ],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                sm_scale=scale,
                logits_soft_cap=logits_soft_cap)
            
            self.normal_wrapper.plan(
                [self.normal_query_start_loc, self.normal_second_level_query_start_loc],
                [self.normal_paged_kv_indptr, self.normal_second_layer_kv_indptr], 
                [self.normal_paged_kv_indices, self.normal_second_layer_kv_indices], 
                [self.normal_paged_kv_last_page_len, self.normal_second_layer_kv_last_page_len], 
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                sm_scale=scale,
                logits_soft_cap=logits_soft_cap
            )


    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('wrapper')
        skip_fields.add('normal_wrapper')
        return super().asdict_zerocopy(skip_fields)

    @property
    def prefill_metadata(self) -> Optional["FlashInferMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_decode_tokens == 0:
            assert self.num_prefills > 0
            return self

        return None

    @property
    def decode_metadata(self) -> Optional["FlashInferMetadata"]:
        # Currently chunked prefill is not supported
        if self.num_prefills > 0:
            assert self.num_decode_tokens == 0, (
                "Chunked prefill is not supported with flashinfer yet.")
            return None

        return self

    def advance_step(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        sampled_token_ids: Optional[torch.Tensor],
        block_size: int,
        num_seqs: int,
        num_queries: int,
    ):
        """
        Update metadata in-place to advance one decode step.
        """

        assert num_seqs > 0
        assert num_queries > 0
        assert model_input.attn_metadata is not None
        assert sampled_token_ids is not None

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        model_input.input_tokens[:num_queries] = sampled_token_ids.flatten()

        # Update GPU tensors
        ops.advance_step_flashinfer(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=model_input.input_tokens,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            block_table_bound=self.block_table_bound)


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)

        # Store the concatenated indices of shared prefix of the requests
        self.paged_kv_indices: List[int] = []
        # Index pointers to the start of each shared blocks
        self.paged_kv_indptr: List[int] = [0]
        # The length of the last page of the shared kvs
        self.paged_kv_last_page_len: List[int] = []

        # Store concatenated page indices of requests for the second layer
        self.second_layer_kv_indices: List[int] = []
        # Index pointers to the start of each request's page indices
        self.second_layer_kv_indptr: List[int] = [0]
        # The length of the last page of each request in the second layer
        self.second_layer_kv_last_page_len: List[int] = []

        self.normal_paged_kv_indices = []
        self.normal_paged_kv_indptr = [0]
        self.normal_paged_kv_last_page_len = []
        self.normal_second_layer_kv_indices = []
        self.normal_second_layer_kv_indptr = [0]
        self.normal_second_layer_kv_last_page_len = []

        self.total_blocks = 0
        self.is_profile_run: bool = False

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, common_prefix):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            is_profile_run = is_block_tables_empty(block_tables)

            # Compute slot mapping.
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            if is_profile_run:
                self.is_profile_run = is_profile_run
                return

            block_table = block_tables[seq_id]
            self._update_unique_kv_tensors(block_table, seq_len, common_prefix)
            self._update_normal_unique_kv_tensors(block_table, seq_len, [])

    def _update_normal_unique_kv_tensors(self, block_table: List[int], seq_len: int, common_prefix: List[int]) -> None:
        shared_length = len(common_prefix)
        # self.total_blocks += (len(block_table) - shared_length)
        block_table_bound = (seq_len) // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else (seq_len) // self.block_size
        self.normal_second_layer_kv_indices.extend(
            block_table[shared_length:block_table_bound])
        self.normal_second_layer_kv_indptr.append(self.normal_second_layer_kv_indptr[-1] +
                                           (block_table_bound - shared_length))
        # FIXME: this calculation will need to be changed
        # once shared level is working
        last_page_len = (seq_len) % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.normal_second_layer_kv_last_page_len.append(last_page_len)

    def _update_unique_kv_tensors(self, block_table: List[int], seq_len: int,
                                  common_prefix: List[int]) -> None:
        """
        Updates the unique level kv tensors
        """
        shared_length = len(common_prefix)
        self.total_blocks += (len(block_table) - shared_length)
        block_table_bound = (seq_len) // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else (seq_len) // self.block_size
        self.second_layer_kv_indices.extend(
            block_table[shared_length:block_table_bound])
        self.second_layer_kv_indptr.append(self.second_layer_kv_indptr[-1] +
                                           (block_table_bound - shared_length))
        # FIXME: this calculation will need to be changed
        # once shared level is working
        last_page_len = (seq_len-16) % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.second_layer_kv_last_page_len.append(last_page_len)

    def _update_shared_kv_tensors(self, common_prefix: List[int],
                                  batch_size: int) -> None:
        """
        Updates the shared level kv tensors
        """
        if not common_prefix:
            # if we don't have common prefixes, we only use the unique level
            # so we fill the first level indices, indptr, last page len with 0s
            # to conform with multilevel wrapper input requirements
            self.paged_kv_indices.extend([0] * batch_size)
            self.paged_kv_indptr.extend([0] * batch_size)
            self.paged_kv_last_page_len.extend([0] * batch_size)
        else:
            # FIXME: this still doesn't work
            self.total_blocks += len(common_prefix)
            self.paged_kv_indices.extend(common_prefix)
            self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                        len(common_prefix))
            self.paged_kv_last_page_len.append(16)
        
    def _update_normal_shared_kv_tensors(self, common_prefix: List[int],
                                  batch_size: int) -> None:
        """
        Updates the shared level kv tensors
        """
        if not common_prefix:
            # if we don't have common prefixes, we only use the unique level
            # so we fill the first level indices, indptr, last page len with 0s
            # to conform with multilevel wrapper input requirements
            self.normal_paged_kv_indices.extend([0] * batch_size)
            self.normal_paged_kv_indptr.extend([0] * batch_size)
            self.normal_paged_kv_last_page_len.extend([0] * batch_size)
    
    def get_shared_blocks_nums(
        self,
        inter_data_list: List["ModelInputForGPUBuilder.InterDataForSeqGroup"]
    ) -> List[int]:
        """
        Returns a list of consecutive shared blocks across sequence groups
        """
        # if len(inter_data_list) == 1:
        #     return []

        flattened_lists = []
        for data in inter_data_list:
            if data.block_tables:
                flattened_lists += list(data.block_tables.values())
        
        if flattened_lists == 1:
            return []

        common_prefix: List[int] = []
        for i, block_tuple in enumerate(zip(*flattened_lists)):
            if all(block == block_tuple[0] for block in block_tuple):
                if i > 0 and block_tuple[0] != common_prefix[-1] + 1:
                    break
                common_prefix.append(block_tuple[0])
            else:
                break

        return common_prefix


    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """

        # FIXME: we set common_prefix to empty list now since
        # shared level is not working yet.
        # is_prompt = all([inter_data.is_prompt for inter_data in self.input_builder.inter_data_list])

        common_prefix = self.get_shared_blocks_nums(
            self.input_builder.inter_data_list
        )

        for inter_data in self.input_builder.inter_data_list:

            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                common_prefix)

        self._update_shared_kv_tensors(common_prefix, len(query_lens))
        self._update_normal_shared_kv_tensors([], len(query_lens))

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            max_blocks = input_block_tables.shape[1]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    num_blocks = len(block_table)
                    if num_blocks <= max_blocks:
                        input_block_tables[i, :num_blocks] = block_table
                    else:
                        # It may be possible to have more blocks allocated due
                        # to lookahead slots of multi-step, however, they are
                        # not used anyway, so can be safely ignored.
                        input_block_tables[
                            i, :max_blocks] = block_table[:max_blocks]

            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)

            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        second_level_query_start_loc = torch.zeros(query_lens_tensor.shape[0] +
                                                   1,
                                                   dtype=torch.int32,
                                                   device=device)

        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=second_level_query_start_loc[1:])

        normal_query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        
        normal_second_level_query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=query_start_loc.dtype,
                         out=normal_query_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                dim=0,
                dtype=query_start_loc.dtype,
                out=normal_second_level_query_start_loc[1:])

        
        if not common_prefix:
            # if no common prefix, we only use the unique kv level, so
            # we just set the first level query start loc the same as
            # the second levels
            torch.cumsum(query_lens_tensor,
                         dim=0,
                         dtype=query_start_loc.dtype,
                         out=query_start_loc[1:])
        else:
            # when we use shared level of the multilevel wrapper
            query_start_loc = torch.tensor(
                [0, second_level_query_start_loc[-1]],
                dtype=torch.int32,
                device=device)

        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            # self.second_layer_kv_indices.extend(
            #     [0] * (self.total_blocks - len(self.second_layer_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)

            second_level_kv_indices_tensor = torch.tensor(
                self.second_layer_kv_indices, device="cpu", dtype=torch.int)
            second_level_kv_indptr_tensor = torch.tensor(
                self.second_layer_kv_indptr, device="cpu", dtype=torch.int)
            second_level_kv_last_page_len_tensor = torch.tensor(
                self.second_layer_kv_last_page_len,
                device="cpu",
                dtype=torch.int)

            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device="cpu",
                                                   dtype=torch.int)
            # Normal wrapper tensors
            normal_paged_kv_indices_tensor = torch.tensor(
                self.normal_paged_kv_indices, device="cpu", dtype=torch.int
            )
            normal_paged_kv_indptr_tensor = torch.tensor(
                self.normal_paged_kv_indptr, device="cpu", dtype=torch.int
            )
            normal_paged_kv_last_page_len_tensor = torch.tensor(
                self.normal_paged_kv_last_page_len, device="cpu", dtype=torch.int
            )

            normal_second_layer_kv_indices_tensor = torch.tensor(
                self.normal_second_layer_kv_indices, device="cpu", dtype=torch.int
            )
            normal_second_layer_kv_indptr_tensor = torch.tensor(
                self.normal_second_layer_kv_indptr, device="cpu", dtype=torch.int
            )
            normal_second_layer_kv_last_page_len_tensor = torch.tensor(
                self.normal_second_layer_kv_last_page_len, device="cpu", dtype=torch.int
            )

        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None
            block_table_bound_tensor = None
            second_level_kv_indices_tensor = None
            second_level_kv_indptr_tensor = None
            second_level_kv_last_page_len_tensor = None

            normal_query_start_loc = None
            normal_second_level_query_start_loc = None
            normal_paged_kv_indices_tensor = None
            normal_paged_kv_indptr_tensor = None
            normal_paged_kv_last_page_len_tensor = None
            normal_second_layer_kv_indices_tensor = None
            normal_second_layer_kv_indptr_tensor = None
            normal_second_layer_kv_last_page_len_tensor = None


        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        return FlashInferMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            max_prefill_seq_len=max_prefill_seq_len,
            block_tables=block_tables,
            paged_kv_indptr=paged_kv_indptr_tensor,
            paged_kv_indices=paged_kv_indices_tensor,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            second_layer_kv_indptr=second_level_kv_indptr_tensor,
            second_layer_kv_indices=second_level_kv_indices_tensor,
            second_layer_kv_last_page_len=second_level_kv_last_page_len_tensor,
            second_level_query_start_loc=second_level_query_start_loc,
            block_table_bound=block_table_bound_tensor,
            seq_lens_tensor=seq_lens_tensor,
            num_qo_heads=self.runner.model_config.get_num_attention_heads(
                self.runner.parallel_config),
            num_kv_heads=self.runner.model_config.get_num_kv_heads(
                self.runner.parallel_config),
            head_dim=self.runner.model_config.get_head_size(),
            page_size=self.block_size,
            seq_start_loc=seq_start_loc,
            query_start_loc=query_start_loc,
            device=device,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            use_cuda_graph=use_captured_graph,
            is_profile_run=self.is_profile_run,

            # Normal wrapper tensors
            normal_paged_kv_indptr=normal_paged_kv_indptr_tensor,
            normal_paged_kv_indices=normal_paged_kv_indices_tensor,
            normal_paged_kv_last_page_len=normal_paged_kv_last_page_len_tensor,
            normal_second_layer_kv_indptr=normal_second_layer_kv_indptr_tensor,
            normal_second_layer_kv_indices=normal_second_layer_kv_indices_tensor,
            normal_second_layer_kv_last_page_len=normal_second_layer_kv_last_page_len_tensor,
            normal_query_start_loc=normal_query_start_loc,
            normal_second_level_query_start_loc=normal_second_level_query_start_loc
            )


class FlashInferImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is not None:
            raise ValueError("Sliding window is not supported in FlashInfer.")
        self.sliding_window = (-1, -1)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: FlashInferMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        assert k_scale == 1.0 and v_scale == 1.0, (
            "key/v_scale is not supported in FlashInfer.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if attn_metadata.num_prefill_tokens > 0:
            assert attn_metadata.num_decode_tokens == 0, (
                "Chunked prefill is not supported with flashinfer yet.")
        if attn_metadata.num_decode_tokens > 0:
            assert attn_metadata.num_prefill_tokens == 0, (
                "Chunked prefill is not supported with flashinfer yet.")

        if kv_cache is not None:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )
            # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
            # to process the cache when the kv_cache_dtype is fp8
            if self.kv_cache_dtype.startswith("fp8"):
                torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                    self.kv_cache_dtype)
                kv_cache = kv_cache.view(torch_dtype)

        query = query.contiguous(
        )  # Flashinfer requires query to be contiguous
        if prefill_meta := attn_metadata.prefill_metadata:
            # We will use flash attention for prefill
            # when kv_cache is not provided.
            # This happens when vllm runs the profiling to
            # determine the number of blocks.
            if kv_cache is None:
                output = torch.ops.vllm.flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                )
            else:
                assert prefill_meta is not None
                assert prefill_meta.wrapper is not None
                output = prefill_meta.wrapper.run(query, kv_cache) ##two level prefill
                output2 = prefill_meta.normal_wrapper.run(query, kv_cache) ## one level prefill
                print(torch.isclose(output, output2, rtol=1e-2, atol=1e-2))

        else:
            assert attn_metadata.decode_metadata is not None
            assert attn_metadata.decode_metadata.wrapper is not None
            output = attn_metadata.decode_metadata.wrapper.run(query, kv_cache) ## two level decode
            output2 = attn_metadata.decode_metadata.normal_wrapper.run(query, kv_cache)
            
            print(torch.isclose(output, output2, rtol=1e-2, atol=1e-2))

        return output.view(num_tokens, hidden_size)
