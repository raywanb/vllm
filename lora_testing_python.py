from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from huggingface_hub import snapshot_download

lora_conf = snapshot_download(repo_id="KarthiAru/peft-lora-starcoder-personal-copilot-A100-40GB-colab")

lora_request=LoRARequest("testing adapter", 1, lora_conf)

llm = LLM(model="bigcode/starcoder", tensor_parallel_size=2, enable_lora=True)