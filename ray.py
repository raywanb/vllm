# %%
import torch
from vllm import LLM, SamplingParams


# %%
import ray
ray.init()

# %%
prompt = "apples are good?"
sampling_params = SamplingParams(temperature=0.8, top_p=0.5, max_tokens=100)
model_path = "raywanb/llama-2-3bit-gptq"
llm = LLM(model=model_path, trust_remote_code=True, quantization="gptq", tokenizer_mode="slow", enforce_eager=True)

# # model_path = "TheBloke/Llama-2-13B-fp16"
# llm = LLM(model=model_path, trust_remote_code=True, tokenizer_mode="slow")

outputs = llm.generate(prompt, sampling_params)


# %%
outputs = llm.generate("mclaren p1", sampling_params)

# %%
print(outputs)

# %%
outputs

# %%



