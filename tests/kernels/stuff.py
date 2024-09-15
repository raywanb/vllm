import pytest
import torch
import flashinfer
from typing import Tuple


@pytest.mark.parametrize("num_heads", [(16, 16)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("seq_lens", [[(2048, 2048)]])
@pytest.mark.parametrize("num_runs", [1000])
@pytest.mark.parametrize("beam_width", [4])
@torch.inference_mode()
def test_flashinfer_batchprefill_beam_search(
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    soft_cap: float,
    seq_lens: list,
    num_runs: int,
    block_size: int,
    beam_width: int,
    query_override: torch.Tensor=None,
    key_value_cache: torch.Tensor=None
) -> None:
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]

    max_num_blocks_per_seq = 1024
    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

    if key_value_cache is None:
        key_value_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda').reshape(num_blocks, 2, block_size, num_kv_heads, head_size)
        
    workspace_size = 128 * 1024 * 1024 
    workspace_buffer_decode = torch.empty(workspace_size, dtype=torch.int8, device='cuda')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_decode, "NHD")


    block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq), dtype=torch.int32)
    block_offset = 0  # This will track the starting point for each sequence's shared blocks

    for start_seq in range(num_seqs):
        shared_len = kv_lens[start_seq] // block_size 

        for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
            block_tables[i, :shared_len] = torch.arange(block_offset, block_offset + shared_len)

        block_offset += shared_len

        for i in range(beam_width):
            beam_index = start_seq * beam_width + i
            unique_start = block_offset + i
            block_tables[beam_index, shared_len:max_num_blocks_per_seq] = torch.arange(
                unique_start, unique_start + (max_num_blocks_per_seq - shared_len) * beam_width, beam_width
            )
        block_offset += (max_num_blocks_per_seq - shared_len) * beam_width


    cumulative_run_time = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    outputs = []

    next_block_index = [(x + block_size - 1) // block_size + 1 for x in kv_lens]  # Index of the next block from block table

    ## REFORMAT KV_INDICES FOR DECODE
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []

    for i in range(num_seqs * beam_width):
        seq_len = kv_lens[i // beam_width]
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.append(list(block_tables[i, :num_blocks]))
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)
        kv_indptr.append(kv_indptr[-1] + num_blocks)

    for step in range(num_runs):
        torch.manual_seed(step)

        query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
            num_seqs * beam_width, num_query_heads, head_size
        )

        kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices_tensor = torch.cat([torch.tensor(x) for x in kv_indices]).reshape(-1)
        kv_last_page_lens_tensor = torch.tensor(kv_last_page_lens, dtype=torch.int32)
        
        decode_wrapper.begin_forward(kv_indptr_tensor, kv_indices_tensor, kv_last_page_lens_tensor, num_query_heads, num_kv_heads, head_size, block_size, "NONE", data_type=dtype)

        start_event.record()
        output = decode_wrapper.forward(
            query,
            key_value_cache,
            "NONE"
        )
        end_event.record()
        torch.cuda.synchronize()
        decode_time = start_event.elapsed_time(end_event)
        cumulative_run_time += decode_time

        outputs.append(output)

        if step % block_size == 0:
            for i in range(beam_width * num_seqs):
                kv_indices[i].append(block_tables[i, next_block_index[i // beam_width]])

            for i in range(len(next_block_index)): 
                next_block_index[i] += 1

            for i in range(1, beam_width * num_seqs + 1):
                kv_indptr[i] += i
        kv_last_page_lens = [(x + 1) % block_size or block_size for x in kv_last_page_lens]

    print(f"NORMAL PREFILL/DECODE: Total cumulative time for .forward calls: {cumulative_run_time} ms")
    return outputs

@pytest.mark.parametrize("num_heads", [(16, 16)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("seq_lens", [[(8192, 8192)]])
@pytest.mark.parametrize("num_runs", [1000])
@pytest.mark.parametrize("beam_width", [4])
@pytest.mark.parametrize("num_levels", [2])
@torch.inference_mode()
def test_multilevel_cascade_attention_wrapper(
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    seq_lens: list,
    num_runs: int,
    block_size: int,
    beam_width: int,
    num_levels: int,
    query_override: torch.Tensor = None,
    key_value_cache: torch.Tensor = None
) -> None:
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

   
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]

    max_num_blocks_per_seq = 1024
    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

    if key_value_cache is None:
        key_value_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')

    workspace_size = 128 * 1024 * 1024
    workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device='cuda')
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(num_levels, workspace_buffer, "NHD")

    block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq), dtype=torch.int32)
    block_offset = 0  # This will track the starting point for each sequence's shared blocks

    for start_seq in range(num_seqs):
        shared_len = kv_lens[start_seq] // block_size 

        for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
            block_tables[i, :shared_len] = torch.arange(block_offset, block_offset + shared_len)

        block_offset += shared_len

        for i in range(beam_width):
            beam_index = start_seq * beam_width + i
            unique_start = block_offset + i
            block_tables[beam_index, shared_len:max_num_blocks_per_seq] = torch.arange(
                unique_start, unique_start + (max_num_blocks_per_seq - shared_len) * beam_width, beam_width
            )
        block_offset += (max_num_blocks_per_seq - shared_len) * beam_width


    qo_indptr_arr = [
        torch.tensor([0, beam_width * num_seqs], dtype=torch.int32, device='cuda'), # this is a bit weird
        torch.arange(beam_width * num_seqs + 1, dtype=torch.int32, device="cuda") 
    ]   

    shared_kv_page_indptr = [0]
    unique_kv_page_indptr = [0]
    shared_kv_page_indices = []
    unique_kv_page_indices = []
    shared_kv_last_page_len = []
    unique_kv_last_page_len = []

    query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
        num_seqs * beam_width, num_query_heads, head_size
    )

    ##Filling the shared metadatas
    for i in range(num_seqs):
        seq_len = kv_lens[i // beam_width]
        num_shared_blocks = (seq_len + block_size - 1) // block_size
        shared_kv_page_indices.append(list(block_tables[i, :num_shared_blocks]))
        shared_kv_page_indptr.append(shared_kv_page_indptr[-1] + num_shared_blocks)
        shared_kv_len = seq_len % block_size
        if shared_kv_len == 0:
            shared_kv_len = block_size
        shared_kv_last_page_len.append(shared_kv_len)

    for i in range(num_seqs * beam_width):
        num_unique_blocks = 0
        unique_kv_page_indices.append([])
        unique_kv_page_indptr.append(unique_kv_page_indptr[-1] + num_unique_blocks)
        unique_kv_last_page_len.append(block_size) ##maybe should be other?

    shared_kv_page_indptr = torch.tensor(shared_kv_page_indptr, dtype=torch.int32, device='cuda')
    shared_kv_page_indices = torch.cat([torch.tensor(x) for x in shared_kv_page_indices]).reshape(-1)
    shared_kv_last_page_len = torch.tensor(shared_kv_last_page_len, dtype=torch.int32, device='cuda')

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    cumulative_run_time = 0.0

    outputs = []

    next_block_index = [(x+block_size-1)//block_size + 1 for x in kv_lens] ## aka index of the next block from block table that we add 

    for step in range(num_runs):
        
        torch.manual_seed(step)

        query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
            num_seqs * beam_width, num_query_heads, head_size
        )

        wrapper.plan(
            qo_indptr_arr,
            [shared_kv_page_indptr,
            torch.tensor(unique_kv_page_indptr, dtype=torch.int32, device='cuda')],
            [shared_kv_page_indices,
            torch.cat([torch.tensor(x) for x in unique_kv_page_indices]).reshape(-1)],
            [shared_kv_last_page_len,
            torch.tensor(unique_kv_last_page_len, dtype=torch.int32, device='cuda')],
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size
        )

        start_event.record()
        output = wrapper.run(query, key_value_cache)
        end_event.record()
        torch.cuda.synchronize()


        cumulative_run_time += start_event.elapsed_time(end_event)

        outputs.append(output)

        if step % block_size == 0:
            for i in range(beam_width * num_seqs):
                unique_kv_page_indices[i].append(block_tables[i, next_block_index[i //beam_width]])
            for i in range(len(next_block_index)):
                next_block_index[i] += 1
            for i in range(1, beam_width * num_seqs + 1):
                unique_kv_page_indptr[i] += i

        unique_kv_last_page_len = [(x + 1) % block_size or block_size for x in unique_kv_last_page_len]

    print("CASCADE: Total cumulative time for .run calls:", cumulative_run_time)
    return outputs


def initialize_key_value_cache(num_seqs, beam_width, max_num_blocks_per_seq, block_size, num_kv_heads, head_size, dtype):
    num_blocks = max_num_blocks_per_seq * num_seqs * beam_width
    key_value_cache = torch.randn(
        num_blocks, 2, block_size, num_kv_heads, head_size, 
        dtype=dtype, 
        device='cuda'
    )
    return key_value_cache


def run_and_compare_flashinfer_tests():
    common_params = {
        "num_heads": (16, 16),
        "head_size": 128,
        "dtype": torch.float16,
        "block_size": 16,
        "seq_lens": [(8192, 8192)],
        "num_runs": 1000,
        "beam_width": 18
    }

    num_seqs = len(common_params["seq_lens"])
    max_num_blocks_per_seq = 1024 
    num_query_heads, num_kv_heads = common_params["num_heads"]

    key_value_cache = initialize_key_value_cache(
        num_seqs, 
        common_params["beam_width"], 
        max_num_blocks_per_seq, 
        common_params["block_size"], 
        num_kv_heads, 
        common_params["head_size"], 
        common_params["dtype"]
    )

    # Run cascade test/
    cascade_outputs = test_multilevel_cascade_attention_wrapper(
        **common_params,
        num_levels=2,
        key_value_cache=key_value_cache
    )

    cascade_outputs = [output.cpu() for output in cascade_outputs]

    # Run batchprefill test
    batchprefill_outputs = test_flashinfer_batchprefill_beam_search(
        **common_params,
        soft_cap=None,
        key_value_cache=key_value_cache
    )

    batchprefill_outputs = [output.cpu() for output in batchprefill_outputs]

    assert len(cascade_outputs) == len(batchprefill_outputs), "Number of outputs mismatch"

    max_diff = 0
    total_elements = 0
    total_diff = 0

    for i, (cascade_output, batchprefill_output) in enumerate(zip(cascade_outputs, batchprefill_outputs)):
        assert cascade_output.shape == batchprefill_output.shape, f"Shape mismatch at step {i}"

        diff = torch.abs(cascade_output - batchprefill_output)

        max_step_diff = torch.max(diff).item()

        max_diff = max(max_diff, max_step_diff)

        total_elements += cascade_output.numel()
        total_diff += torch.sum(diff).item()

    avg_diff = total_diff / total_elements

    print(f"\nComparison results:")
    print(f"Max absolute difference across all steps: {max_diff}")
    print(f"Average absolute difference: {avg_diff}")

    assert max_diff < 1e-3, f"Max absolute difference ({max_diff}) exceeds threshold"
    assert avg_diff < 1e-4, f"Average absolute difference ({avg_diff}) exceeds threshold"

    print("All tests passed successfully!")

if __name__ == "__main__":
    run_and_compare_flashinfer_tests()



# import pytest
# import torch
# import flashinfer
# from typing import Tuple
# import gc

# def initialize_key_value_cache(num_seqs, beam_width, max_num_blocks_per_seq, block_size, num_kv_heads, head_size, dtype):
#     num_blocks = max_num_blocks_per_seq * num_seqs * beam_width
#     key_value_cache = torch.randn(
#         num_blocks, 2, block_size, num_kv_heads, head_size, 
#         dtype=dtype, 
#         device='cuda'
#     )
#     return key_value_cache

# @pytest.mark.parametrize("num_heads", [(16, 16)])
# @pytest.mark.parametrize("head_size", [128])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("block_size", [16])
# @pytest.mark.parametrize("seq_lens", [[(8192, 8192)]])
# @pytest.mark.parametrize("num_runs", [1000])
# @pytest.mark.parametrize("beam_width", [4])
# @pytest.mark.parametrize("num_levels", [2])
# @torch.inference_mode()
# def test_multilevel_cascade_attention_wrapper(
#     num_heads: Tuple[int, int],
#     head_size: int,
#     dtype: torch.dtype,
#     seq_lens: list,
#     num_runs: int,
#     block_size: int,
#     beam_width: int,
#     num_levels: int,
#     query_override: torch.Tensor = None,
#     key_value_cache: torch.Tensor = None
# ) -> None:
#     torch.set_default_device("cuda")
#     torch.cuda.manual_seed_all(0)

#     num_query_heads, num_kv_heads = num_heads
#     assert num_query_heads % num_kv_heads == 0

   
#     num_seqs = len(seq_lens)
#     query_lens = [x[0] for x in seq_lens]
#     kv_lens = [x[1] for x in seq_lens]

#     max_num_blocks_per_seq = 1024
#     num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

#     # key_value_cache = torch.arange(
#     #     num_blocks * 2 * block_size * num_kv_heads * head_size, 
#     #     dtype=dtype, 
#     #     device='cuda'
#     # ).reshape(num_blocks, 2, block_size, num_kv_heads, head_size)

#     if key_value_cache is None:
#         key_value_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda')

#     workspace_size = 128 * 1024 * 1024
#     workspace_buffer = torch.empty(workspace_size, dtype=torch.uint8, device='cuda')
#     wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(num_levels, workspace_buffer, "NHD")



#     block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq), dtype=torch.int32)
#     block_offset = 0  # This will track the starting point for each sequence's shared blocks

#     for start_seq in range(num_seqs):
#         shared_len = kv_lens[start_seq] // block_size 

#         for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
#             block_tables[i, :shared_len] = torch.arange(block_offset, block_offset + shared_len)

#         block_offset += shared_len

#         for i in range(beam_width):
#             beam_index = start_seq * beam_width + i
#             unique_start = block_offset + i
#             block_tables[beam_index, shared_len:max_num_blocks_per_seq] = torch.arange(
#                 unique_start, unique_start + (max_num_blocks_per_seq - shared_len) * beam_width, beam_width
#             )
#         block_offset += (max_num_blocks_per_seq - shared_len) * beam_width


#     qo_indptr_arr = [
#         torch.tensor([0, beam_width * num_seqs], dtype=torch.int32, device='cuda'), # this is a bit weird
#         torch.arange(beam_width * num_seqs + 1, dtype=torch.int32, device="cuda") 
#     ]   

#     shared_kv_page_indptr = [0]
#     unique_kv_page_indptr = [0]
#     shared_kv_page_indices = []
#     unique_kv_page_indices = []
#     shared_kv_last_page_len = []
#     unique_kv_last_page_len = []

#     query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
#         num_seqs * beam_width, num_query_heads, head_size
#     )


#     ##Filling the shared metadatas
#     for i in range(num_seqs):
#         seq_len = kv_lens[i // beam_width]
#         num_shared_blocks = (seq_len + block_size - 1) // block_size
#         shared_kv_page_indices.append(list(block_tables[i, :num_shared_blocks]))
#         shared_kv_page_indptr.append(shared_kv_page_indptr[-1] + num_shared_blocks)
#         shared_kv_len = seq_len % block_size
#         if shared_kv_len == 0:
#             shared_kv_len = block_size
#         shared_kv_last_page_len.append(shared_kv_len)

#     for i in range(num_seqs * beam_width):
#         num_unique_blocks = 0
#         unique_kv_page_indices.append([])
#         unique_kv_page_indptr.append(unique_kv_page_indptr[-1] + num_unique_blocks)
#         unique_kv_last_page_len.append(block_size) ##maybe should be other?

#     shared_kv_page_indptr = torch.tensor(shared_kv_page_indptr, dtype=torch.int32, device='cuda')
#     shared_kv_page_indices = torch.cat([torch.tensor(x) for x in shared_kv_page_indices]).reshape(-1)
#     shared_kv_last_page_len = torch.tensor(shared_kv_last_page_len, dtype=torch.int32, device='cuda')

#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)

#     cumulative_run_time = 0.0

#     wrapper.plan(
#         qo_indptr_arr,
#         [shared_kv_page_indptr,
#          torch.tensor(unique_kv_page_indptr, dtype=torch.int32, device='cuda')],
#         [shared_kv_page_indices,
#          torch.cat([torch.tensor(x) for x in unique_kv_page_indices]).reshape(-1)],
#         [shared_kv_last_page_len,
#          torch.tensor(unique_kv_last_page_len, dtype=torch.int32, device='cuda')],
#         num_query_heads,
#         num_kv_heads,
#         head_size,
#         block_size
#     )

#     # print("QO_INDPTR", qo_indptr_arr)
#     # print("SHARED KV INDPTR", shared_kv_page_indptr)
#     # print("SHARED KV INDICES", shared_kv_page_indices)
#     # print("UNIQUE KV INDPTR", unique_kv_page_indptr)
#     # print("UNIQUE INDICES", unique_kv_page_indices)
#     # print("SHARED PAGED LAST LEN", shared_kv_last_page_len)
#     # print("UNIQUE PAGE LAST LEN", unique_kv_last_page_len)

#     start_event.record()
#     prefill_output = wrapper.run(query, key_value_cache)
#     end_event.record()
#     torch.cuda.synchronize()
#     outputs = [prefill_output]

#     print("CASCADE ", prefill_output[0])

#     cumulative_run_time += start_event.elapsed_time(end_event)

#     next_block_index = [(x+block_size-1)//block_size + 1 for x in kv_lens] ## aka index of the next block from block table that we add 

#     for step in range(num_runs):
#         query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
#             num_seqs * beam_width, num_query_heads, head_size
#         )

#         wrapper.plan(
#             qo_indptr_arr,
#             [shared_kv_page_indptr,
#             torch.tensor(unique_kv_page_indptr, dtype=torch.int32, device='cuda')],
#             [shared_kv_page_indices,
#             torch.cat([torch.tensor(x) for x in unique_kv_page_indices]).reshape(-1)],
#             [shared_kv_last_page_len,
#             torch.tensor(unique_kv_last_page_len, dtype=torch.int32, device='cuda')],
#             num_query_heads,
#             num_kv_heads,
#             head_size,
#             block_size
#         )

#         start_event.record()
#         output = wrapper.run(query, key_value_cache)
#         end_event.record()
#         torch.cuda.synchronize()

#         if step == 0:
#             print(output)

#         cumulative_run_time += start_event.elapsed_time(end_event)

#         outputs.append(output)

#         if step % block_size == 0:
#             for i in range(beam_width * num_seqs):
#                 unique_kv_page_indices[i].append(block_tables[i, next_block_index[i //beam_width]])
#             for i in range(len(next_block_index)):
#                 next_block_index[i] += 1
#             for i in range(1, beam_width * num_seqs + 1):
#                 unique_kv_page_indptr[i] += i

#         unique_kv_last_page_len = [(x + 1) % block_size or block_size for x in unique_kv_last_page_len]
#         # print(output.shape)

#     print("CASCADE: Total cumulative time for .run calls:", cumulative_run_time)
#     return outputs

# @pytest.mark.parametrize("num_heads", [(16, 16)])
# @pytest.mark.parametrize("head_size", [128])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("block_size", [16])
# @pytest.mark.parametrize("soft_cap", [None])
# @pytest.mark.parametrize("seq_lens", [[(2048, 2048)]])
# @pytest.mark.parametrize("num_runs", [1000])
# @pytest.mark.parametrize("beam_width", [4])
# @torch.inference_mode()
# def test_flashinfer_batchprefill_beam_search(
#     num_heads: Tuple[int, int],
#     head_size: int,
#     dtype: torch.dtype,
#     soft_cap: float,
#     seq_lens: list,
#     num_runs: int,
#     block_size: int,
#     beam_width: int,
#     query_override: torch.Tensor=None,
#     key_value_cache: torch.Tensor=None
# ) -> None:
#     torch.set_default_device("cuda")
#     torch.cuda.manual_seed_all(0)

#     num_query_heads, num_kv_heads = num_heads
#     assert num_query_heads % num_kv_heads == 0

#     num_seqs = len(seq_lens)
#     query_lens = [x[0] for x in seq_lens]
#     kv_lens = [x[1] for x in seq_lens]

#     max_num_blocks_per_seq = 1024
#     num_blocks = max_num_blocks_per_seq * num_seqs * beam_width

#     if key_value_cache is None:
#         key_value_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda').reshape(num_blocks, 2, block_size, num_kv_heads, head_size)

#     # key_value_cache = torch.arange(
#     #     num_blocks * 2 * block_size * num_kv_heads * head_size, 
#     #     dtype=dtype, 
#     #     device='cuda'
#     # ).reshape(num_blocks, 2, block_size, num_kv_heads, head_size)
#     workspace_size = 128 * 1024 * 1024 
#     workspace_buffer_prefill = torch.empty(workspace_size, dtype=torch.int8, device='cuda')
#     prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer_prefill, "NHD")

#     workspace_buffer_decode = torch.empty(workspace_size, dtype=torch.int8, device='cuda')
#     decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_decode, "NHD")


#     block_tables = torch.zeros((num_seqs * beam_width, max_num_blocks_per_seq), dtype=torch.int32)
#     block_offset = 0  # This will track the starting point for each sequence's shared blocks

#     for start_seq in range(num_seqs):
#         shared_len = kv_lens[start_seq] // block_size 

#         for i in range(start_seq * beam_width, (start_seq + 1) * beam_width):
#             block_tables[i, :shared_len] = torch.arange(block_offset, block_offset + shared_len)

#         block_offset += shared_len

#         for i in range(beam_width):
#             beam_index = start_seq * beam_width + i
#             unique_start = block_offset + i
#             block_tables[beam_index, shared_len:max_num_blocks_per_seq] = torch.arange(
#                 unique_start, unique_start + (max_num_blocks_per_seq - shared_len) * beam_width, beam_width
#             )
#         block_offset += (max_num_blocks_per_seq - shared_len) * beam_width


#     qo_indptr = [0]
#     kv_indptr = [0]
#     kv_indices = []
#     kv_last_page_lens = []

#     # query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
#     #     num_seqs * beam_width, num_query_heads, head_size
#     # )
#     total_length = sum(query_lens)
#     num_elements = total_length * num_query_heads * head_size

#     # Create a tensor with arange and reshape it to the desired shape
#     query = torch.arange(num_elements, dtype=dtype).reshape(total_length, num_query_heads, head_size)
#     # query = torch.randn(sum(query_lens),
#     #                         num_query_heads,
#     #                         head_size,
#     #                         dtype=dtype)

#     for i in range(num_seqs):
#         seq_len = kv_lens[i]
#         num_blocks = (seq_len + block_size - 1) // block_size
#         qo_indptr.append(qo_indptr[-1] + query_lens[i])
#         kv_indices.extend(block_tables[i, :num_blocks])
#         kv_last_page_len = seq_len % block_size
#         if kv_last_page_len == 0:
#             kv_last_page_len = block_size
#         kv_last_page_lens.append(kv_last_page_len)
#         kv_indptr.append(kv_indptr[-1] + num_blocks)
 

#     qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32)
#     kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32)
#     kv_indices_tensor = torch.tensor(kv_indices).reshape(-1)
#     kv_last_page_lens_tensor = torch.tensor(kv_last_page_lens, dtype=torch.int32)

#     # print(kv_indices_tensor)
#     # print(kv_indptr)
#     # print(qo_indptr)
#     # # print(key_value_cache)
#     cumulative_run_time = 0.0
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     print(block_tables)
#     print(kv_indptr_tensor)
#     print(kv_indices_tensor)
#     print(kv_last_page_lens_tensor)
#     ## PREFILL
#     prefill_wrapper.begin_forward(
#         qo_indptr_tensor,
#         kv_indptr_tensor,
#         kv_indices_tensor,
#         kv_last_page_lens_tensor,
#         num_query_heads,
#         num_kv_heads,
#         head_size,
#         block_size,
#     )
    
#     start_event.record()
#     output = prefill_wrapper.forward(
#         query,
#         key_value_cache,
#         causal=True
#     )

#     outputs = [output]
#     print("PREFILL output", output[-1])
#     end_event.record()
#     torch.cuda.synchronize()
#     prefill_time = start_event.elapsed_time(end_event)
#     cumulative_run_time += prefill_time


#     next_block_index = [(x + block_size - 1) // block_size + 1 for x in kv_lens]  # Index of the next block from block table

#     ## REFORMAT KV_INDICES FOR DECODE
#     kv_indices = []
#     kv_indptr = [0]
#     kv_last_page_lens = []

#     for i in range(num_seqs * beam_width):
#         seq_len = kv_lens[i // beam_width]
#         num_blocks = (seq_len + block_size - 1) // block_size
#         kv_indices.append(list(block_tables[i, :num_blocks]))
#         kv_last_page_len = seq_len % block_size
#         if kv_last_page_len == 0:
#             kv_last_page_len = block_size
#         kv_last_page_lens.append(kv_last_page_len)
#         kv_indptr.append(kv_indptr[-1] + num_blocks)

#     for step in range(num_runs):
#         query = torch.arange(num_seqs * beam_width * num_query_heads * head_size, dtype=dtype, device='cuda').reshape(
#             num_seqs * beam_width, num_query_heads, head_size
#         )

#         kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32)
#         kv_indices_tensor = torch.cat([torch.tensor(x) for x in kv_indices]).reshape(-1)
#         kv_last_page_lens_tensor = torch.tensor(kv_last_page_lens, dtype=torch.int32)
        
#         decode_wrapper.begin_forward(kv_indptr_tensor, kv_indices_tensor, kv_last_page_lens_tensor, num_query_heads, num_kv_heads, head_size, block_size, "NONE", data_type=dtype)

#         start_event.record()
#         output = decode_wrapper.forward(
#             query,
#             key_value_cache,
#             "NONE"
#         )
#         end_event.record()
#         torch.cuda.synchronize()
#         decode_time = start_event.elapsed_time(end_event)
#         cumulative_run_time += decode_time

#         outputs.append(output)

#         if step == 0:
#             print("DECODE step 0", output[0])

#         if step % block_size == 0:
#             for i in range(beam_width * num_seqs):
#                 kv_indices[i].append(block_tables[i, next_block_index[i // beam_width]])

#             for i in range(len(next_block_index)): 
#                 next_block_index[i] += 1

#             for i in range(1, beam_width * num_seqs + 1):
#                 kv_indptr[i] += i
#         kv_last_page_lens = [(x + 1) % block_size or block_size for x in kv_last_page_lens]

#     print(f"NORMAL PREFILL/DECODE: Total cumulative time for .forward calls: {cumulative_run_time} ms")
#     return outputs


# def run_and_compare_flashinfer_tests():
#     common_params = {
#         "num_heads": (16, 16),
#         "head_size": 128,
#         "dtype": torch.float16,
#         "block_size": 16,
#         "seq_lens": [(8192, 8192)],
#         "num_runs": 1000,
#         "beam_width": 4
#     }

#     num_seqs = len(common_params["seq_lens"])
#     max_num_blocks_per_seq = 1024 
#     num_query_heads, num_kv_heads = common_params["num_heads"]

#     key_value_cache = initialize_key_value_cache(
#         num_seqs, 
#         common_params["beam_width"], 
#         max_num_blocks_per_seq, 
#         common_params["block_size"], 
#         num_kv_heads, 
#         common_params["head_size"], 
#         common_params["dtype"]
#     )

#     # Run cascade test/
#     cascade_outputs = test_multilevel_cascade_attention_wrapper(
#         **common_params,
#         num_levels=2,
#         key_value_cache=key_value_cache
#     )

#     cascade_outputs = [output.cpu() for output in cascade_outputs]


#     # Run batchprefill test
#     batchprefill_outputs = test_flashinfer_batchprefill_beam_search(
#         **common_params,
#         soft_cap=None,
#         # key_value_cache=key_value_cache
#     )

#     batchprefill_outputs = [output.cpu() for output in batchprefill_outputs]

#     assert len(cascade_outputs) == len(batchprefill_outputs), "Number of outputs mismatch"

#     max_diff = 0
#     total_elements = 0
#     total_diff = 0

#     for i, (cascade_output, batchprefill_output) in enumerate(zip(cascade_outputs, batchprefill_outputs)):
#         assert cascade_output.shape == batchprefill_output.shape, f"Shape mismatch at step {i}"

#         diff = torch.abs(cascade_output - batchprefill_output)

#         max_step_diff = torch.max(diff).item()

#         max_diff = max(max_diff, max_step_diff)

#         total_elements += cascade_output.numel()
#         total_diff += torch.sum(diff).item()

#     avg_diff = total_diff / total_elements

#     print(f"\nComparison results:")
#     print(f"Max absolute difference across all steps: {max_diff}")
#     print(f"Average absolute difference: {avg_diff}")

#     # You can adjust these thresholds based on your specific requirements
#     assert max_diff < 1e-3, f"Max absolute difference ({max_diff}) exceeds threshold"
#     assert avg_diff < 1e-4, f"Average absolute difference ({avg_diff}) exceeds threshold"

#     print("All tests passed successfully!")

# if __name__ == "__main__":
#     run_and_compare_flashinfer_tests()
# # if __name__ == "__main__":
#     pytest.main([__file__, "-s"])
