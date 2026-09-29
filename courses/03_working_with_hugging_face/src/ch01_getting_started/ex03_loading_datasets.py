"""
Chapter 1 - Exercise 3: Loading datasets from the Hub
"""
from datasets import load_dataset
from common.config import HF_TOKEN


def main():
    print("Loading dataset from Hugging Face Hub...")

    # Example using a standard sentiment dataset often covered in introductory modules
    dataset = load_dataset("imdb", split="train[:100]", token=HF_TOKEN)

    print(f"Dataset schema:\n{dataset}")
    print("\nFirst sample preview:")
    print(f"Text: {dataset[0]['text'][:150]}...")
    print(f"Label: {dataset[0]['label']}")


if __name__ == "__main__":
    main()