from typing import List, Optional, Tuple

import flashinfer
import pytest
import torch
import time
from itertools import accumulate

NUM_HEADS = [(16, 16), (32, 8)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len -
                                             (query_len + sliding_window) +
                                             1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
@torch.inference_mode
def test_flashinfer_decode_with_paged_kv(kv_lens: List[int],
                                         num_heads: Tuple[int,
                                                          int], head_size: int,
                                         dtype: torch.dtype, block_size: int,
                                         soft_cap: Optional[float]) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.\
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.begin_forward(kv_indptr,
                          kv_indices,
                          kv_last_page_lens,
                          num_query_heads,
                          num_kv_heads,
                          head_size,
                          block_size,
                          "NONE",
                          data_type=dtype)

    output = wrapper.forward(query, key_value_cache, logits_soft_cap=soft_cap)

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=[1] * num_seqs,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 30.0, 50.0])
@torch.inference_mode
def test_flashinfer_prefill_with_paged_kv(seq_lens: List[Tuple[int, int]],
                                          num_heads: Tuple[int, int],
                                          head_size: int, dtype: torch.dtype,
                                          block_size: int,
                                          soft_cap: Optional[float]) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD")
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
    )

    output = wrapper.forward(
        query,
        key_value_cache,
        logits_soft_cap=soft_cap,
    )

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                soft_cap=soft_cap)
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - ref_output))}"

@torch.inference_mode
def test_flashinfer_prefill_unshared(seq_lens: List[Tuple[int, int]],
                                          num_heads: Tuple[int, int],
                                          head_size: int, dtype: torch.dtype,
                                          block_size: int,
                                          soft_cap: Optional[float]) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.arange(max_num_blocks_per_seq*num_seqs, dtype=torch.int32).reshape(num_seqs, max_num_blocks_per_seq)


    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])
        
    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD")
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    output = wrapper.forward(
        query,
        key_value_cache,
        logits_soft_cap=soft_cap,
    )
    torch.cuda.synchronize()
    unshared_time = (time.perf_counter() - start_time) / 10
    # ref_output = ref_paged_attn(query=query,
    #                             key_cache=key_cache,
    #                             value_cache=value_cache,
    #                             query_lens=query_lens,
    #                             kv_lens=kv_lens,
    #                             block_tables=block_tables,
    #                             scale=scale,
    #                             soft_cap=soft_cap)
    
    return output, unshared_time

@torch.inference_mode
def test_flashinfer_prefill_shared(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int, dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float]) -> None:

    torch.set_default_device('cuda')
    torch.cuda.manual_seed_all(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]

    assert num_query_heads % num_kv_heads == 0 

    max_kv_len = max(kv_lens)
    scale = head_size ** -0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)

    shared_prefix_len = 32768
    unique_suffix_len = max(0, max_kv_len - shared_prefix_len)
    total_seq_len = shared_prefix_len + unique_suffix_len
    max_num_blocks_per_seq = (total_seq_len + block_size - 1) // block_size

    shared_blocks = (shared_prefix_len + block_size - 1) // block_size
    unique_blocks = (unique_suffix_len + block_size - 1) // block_size

    key_value_cache = torch.randn(NUM_BLOCKS,
                                  2,
                                  block_size,
                                  num_kv_heads,
                                  head_size,
                                  dtype=dtype)
    
    key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
    value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

    # Normalize the scale of the key and value caches to mitigate
    # numerical instability.
    key_cache /= head_size**0.5
    value_cache /= head_size**0.5

    block_tables = torch.zeros((num_seqs, max_num_blocks_per_seq), dtype=torch.int32)
    for i in range(num_seqs):
        block_tables[i, :shared_blocks] = torch.arange(shared_blocks)
        if unique_blocks > 0:
            block_tables[i, shared_blocks:shared_blocks+unique_blocks] = torch.arange(shared_blocks, shared_blocks + unique_blocks) + i * unique_blocks

    qo_indptr = [0]
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    
    
    for i in range(num_seqs):
        if unique_blocks > 0:
            kv_indices.extend(range(unique_blocks))
        kv_indptr.append(kv_indptr[-1] + unique_blocks)
        
        kv_last_page_len = unique_suffix_len % block_size
        if kv_last_page_len == 0 and unique_suffix_len > 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        qo_indptr.append(qo_indptr[-1] + query_lens[i])

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
        workspace_buffer, "NHD")
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
    )

    k_shared = key_cache[:shared_blocks].reshape(-1, num_kv_heads, head_size)
    v_shared = value_cache[:shared_blocks].reshape(-1, num_kv_heads, head_size)

    unique_kv_cache = key_value_cache[shared_blocks:]

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    output = wrapper.forward(
        query,
        k_shared,
        v_shared,
        unique_kv_cache=unique_kv_cache,
    )
    torch.cuda.synchronize()
    shared_time = (time.perf_counter() - start_time) / 10

    return output, shared_time

def run_tests():
    seq_lens = [(2000, 32868), (1000, 32868), (1000, 32868),(2000, 32868), (1000, 32868), (1000, 32868)]
    num_heads = (32, 32) 
    head_size = 128 
    block_size = 16
    dtype = torch.float16  
    soft_cap = None

    paged_kv_output = test_flashinfer_prefill_with_paged_kv(
        seq_lens, num_heads, head_size, dtype, block_size, soft_cap
    )

    shared_output = test_flashinfer_prefill_shared(
        seq_lens, num_heads, head_size, dtype, block_size, soft_cap
    )

    return paged_kv_output, shared_output

def compare_outputs(paged_kv_output, shared_output):
    shape_match = paged_kv_output[0].shape == shared_output[0].shape
    
    if shape_match:
        max_diff = torch.max(torch.abs(paged_kv_output[0] - shared_output[0]))
        mean_diff = torch.mean(torch.abs(paged_kv_output[0] - shared_output[0]))
        are_close = torch.allclose(paged_kv_output[0], shared_output[0], atol=1e-2, rtol=1e-2)
    else:
        max_diff = None
        mean_diff = None
        are_close = False

    return {
        "shape_match": shape_match,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "are_close": are_close,
        "speedup": paged_kv_output[1]/shared_output[1]
    }

if __name__ == "__main__":
    paged_kv_output, shared_output = run_tests()
    comparison_results = compare_outputs(paged_kv_output, shared_output)
    
    print("Comparison Results:")
    print(f"Shapes match: {comparison_results['shape_match']}")
    print(f"Maximum difference: {comparison_results['max_difference']}")
    print(f"Mean difference: {comparison_results['mean_difference']}")
    print(f"Outputs are close (within 1e-2 tolerance): {comparison_results['are_close']}")
    print(f"Speedup: {comparison_results['speedup']}")