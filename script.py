from vllm import LLM

llm = LLM(model="raywanb/Qwen3-SwiftKV-1.7b", enforce_eager=True)

print(llm.generate("Hello This is a test on Qwen 2.5."))