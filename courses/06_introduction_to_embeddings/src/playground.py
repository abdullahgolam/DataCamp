from openai import OpenAI
from common.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text.")


response_dict = response.model.to_dict()
print(response_dict)