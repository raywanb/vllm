from vllm import LLM

llm = LLM(model="Qwen/Qwen3-1.7b")

print(llm.generate("Hello"))