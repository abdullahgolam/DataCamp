from openai import OpenAI
import numpy as np
from common.config import OPENAI_API_KEY
from common.helpers import summarize_embeddings

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to embed text
def embed_text(text):
    response = client.embeddings.create(input=text, model='text-embedding-ada-002')
    return response.data[0].embedding

# Example usage
text_embedding = embed_text("Sample text for embedding")

print(summarize_embeddings(text_embedding))