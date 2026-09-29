"""
Chapter 1 - Exercise 1: Building a text generation pipeline
"""
from transformers import pipeline
from common.config import DEVICE, HF_TOKEN

def main():
    print(f"Running on device: {DEVICE}")

    # Initialize text generation pipeline with a lightweight model
    generator = pipeline(
        task="text-generation",
        model="gpt2",
        device=0 if DEVICE.type == "cuda" else -1,
        token=HF_TOKEN,
    )

    prompt = "Artificial Intelligence developers typically start by"
    outputs = generator(prompt, max_new_tokens=30, num_return_sequences=1)

    print("\n--- Generated Output ---")
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()