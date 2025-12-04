#!/usr/bin/env python3
"""
Load and test the SwiftKV model with kv_sharing_map support
"""

from vllm import LLM, SamplingParams

def load_swiftkv_model():
    """Load the SwiftKV model from local checkpoint"""
    
    # Path to your trained model checkpoint
    model_path = "/data/raywanb/checkpoint/qwen3-swiftkv-8b/global_step_1234"
    # model_path = "Qwen/Qwen3-8B"
    
    print(f"Loading SwiftKV model from: {model_path}")
    
    # Initialize the LLM with the local model path
    llm = LLM(
        model=model_path,
        trust_remote_code=True,  # Required for custom SwiftKV models
        max_model_len=40960,  # Match max_position_embeddings
        tensor_parallel_size=1,  # Adjust based on your GPU setup
    )
    
    print("✅ Model loaded successfully!")
    
    # # Test the model with a simple prompt
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=100,
    )
    
    prompts = [
        "Hello! How are you today?",
        "Explain what SwiftKV is:",
        "What is the capital of France? Answer in 10 or more words."
    ]
    
    print("\n🧪 Testing the model...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Response: {generated_text}")
        print("-" * 50)
    
    return llm

if __name__ == "__main__":
    try:
        llm = load_swiftkv_model()
        print("\n🎉 SwiftKV model test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        raise
