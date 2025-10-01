# run_qwen3_17b_bench.py
# Usage (examples at bottom):
#   python run_qwen3_17b_bench.py
#   python run_qwen3_17b_bench.py --num-prompts 2000 --input-len 512 --output-len 256
#   python run_qwen3_17b_bench.py --tensor-parallel-size 2

import argparse
from vllm.benchmarks import throughput as bench

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Qwen3-1.7B offline throughput (vLLM)")
    # --- core model/backend settings ---
    p.add_argument("--model", type=str, default="raywanb/Qwen3-SwiftKV-1.7B",
                   help="HF model id")
    p.add_argument("--backend", type=str, default="vllm",
                   choices=["vllm", "hf", "mii", "vllm-chat"])
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Set if the model repo needs it (Qwen often okay without).")

    # --- dataset knobs (ShareGPT-like by default) ---
    p.add_argument("--dataset-name", type=str, default="sharegpt",
                   choices=["sharegpt","random","sonnet","burstgpt","hf","prefix_repetition"])
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--num-prompts", type=int, default=1000)
    p.add_argument("--input-len", type=int, default=256,
                   help="Used by random/sonnet; ignored by sharegpt.")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--enforce-eager", type=bool, default=True)

    # --- vLLM engine performance knobs (forwarded via EngineArgs.add_cli_args) ---
    # We piggyback on vLLM’s own CLI arg builder so you can pass common flags like:
    # --tensor-parallel-size, --max-model-len, --gpu-memory-utilization, --kv-cache-dtype, etc.
    bench.add_cli_args(p)

    # sensible defaults
    p.set_defaults(
        n=1,                          # 1 sample per prompt
        seed=0,                       # reproducibility
        disable_detokenize=False,     # include detokenization time
        profile=False,                # torch profiler off
        async_engine=False,           # use LLM() path for offline throughput
        tokenizer=None,               # default to model
        output_json="qwen3_17b_throughput.json",
    )
    return p.parse_args()

def main():
    args = build_args()

    # Force our model unless the caller overrides --model
    if not getattr(args, "model", None):
        args.model = "raywanb/Qwen3-SwiftKV-1.7B"

    # A couple of handy defaults for smaller GPUs (comment out if you don’t need them):
    if not getattr(args, "max_model_len", None):
        # Qwen3-1.7B supports long context, but smaller max_model_len saves memory
        args.max_model_len = 8192
    if not getattr(args, "gpu_memory_utilization", None):
        args.gpu_memory_utilization = 0.95

    # Run the standard vLLM offline throughput benchmark
    bench.main(args)

if __name__ == "__main__":
    main()
