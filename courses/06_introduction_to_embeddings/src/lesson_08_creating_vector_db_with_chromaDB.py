"""
Installing ChromaDB
    - ChromaDB is a simple yet powerful vector database
    - Two flavors:
        - Local: for development and prototyping
        - Client/Server: made for production
"""
from openai import OpenAI

"""
Connecting to the database
we first need to create a client. We import chromadb, and create a persistent client by calling PersistentClient. Persistent clients save the database files to disk at the path specified.
"""
from pprint import pprint
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from common.config import OPENAI_API_KEY

client = chromadb.PersistentClient(path=r'D:\chromadb')

"""
To add embeddings to the database, we must first create a collection. Collections are analogous to tables, where we can create as many as we want to store our data. To create the collection we use the .create_collection() method by passing the name of our collection. and the function for creating the embeddings: here, we specify the OpenAI embedding function and API key. In Chroma, and on many other vector databases, a default embedding function is used automatically if one isn't specified.
"""
# collection = client.create_collection(
#     name='my_collection',
#     embedding_function=OpenAIEmbeddingFunction(
#         model_name="text-embedding-3-small",
#         api_key=OPENAI_API_KEY
#     )
# )

"""
the list_collections() method returns a list of all the collections in the database.
[Collection(name=my_collection)]
"""
# print(client.list_collections())

"""
Inserting embeddings
    - Single document
    - IDs must be provided
    - Embeddings will be created by the collection!
"""
# collection.add(ids=["my-doc-1"], documents=["This is the first source text"])

"""
Inserting embeddings
    - Multiple documents
"""
# collection.add(
#     ids=["my-doc-2", "my-doc-3"],
#     documents=["This is the second source text", "This is the third source text"]
# )

"""
return the total number of documents
"""
# print(client.count_collections())

"""
return the first ten items in the collection.
The collection already exists on disk, so we fetch it with get_collection() instead of creating it again.
"""
collection = client.get_collection(
    name='my_collection',
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)
# print(collection.peek())

"""
return particular documents by their IDs.
"""
# pprint(collection.get(ids=["my-doc-1"]))

"""
Estimating embedding cost
- Embedding model (text-embedding-3-small) costs $0.00002 per 1,000 tokens.
- To calculate:
    cost = 0.00002 * len(tokens)/1000
- Count tokens with the tiktoken library
    pip install tiktoken
    tiktoken can convert any text into tokens. First we use the encoding_for_model function to get a token encoder for embedding model we're using. To calculate the total number of tokens, we use the following Python code.
"""
import tiktoken

documents = collection.peek(limit=3)
enc = tiktoken.encoding_for_model("text-embedding-3-small")
total_tokens = sum(len(enc.encode(text)) for text in documents)

cost_per_1k_tokens = 0.00002

print("Total tokens:", total_tokens)
print("Cost:", cost_per_1k_tokens * total_tokens)