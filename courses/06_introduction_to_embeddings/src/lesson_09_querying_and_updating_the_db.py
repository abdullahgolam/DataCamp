"""
Querying and updating the database
    - Create a collection named netflix_titles for the titles in Data/netflix_titles_1000.csv
"""
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from common.config import OPENAI_API_KEY

DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "netflix_titles_1000.csv"

client = chromadb.PersistentClient(path=r'D:\chromadb')

"""
get_or_create_collection() returns the collection if it already exists, so the script can be re-run safely.
"""
# collection = client.get_or_create_collection(
#     name='netflix_titles',
#     embedding_function=OpenAIEmbeddingFunction(
#         model_name="text-embedding-3-small",
#         api_key=OPENAI_API_KEY
#     )
# )
# print(client.list_collections())

"""
Prepare the data to insert: the show_id column becomes the IDs, and each title's text becomes a document.
"""
# df = pd.read_csv(DATA_PATH)
#
# ids = df['show_id'].tolist()
# documents = [
#     f"Title: {row.title} ({row.type})\nDescription: {row.description}\nCategories: {row.listed_in}"
#     for row in df.itertuples()
# ]

"""
Inserting the documents (not run yet). Each add() call creates the embeddings through the OpenAI API.
"""
# collection.add(ids=ids, documents=documents)
# print("Documents in collection:", collection.count())

"""
The collection is still empty, so preview the first 10 prepared records instead.
"""
# for id_, document in zip(ids[:10], documents[:10]):
#     print(f"{id_}\n{document}\n")

"""
After inserting, peek() returns the first 10 records stored in the collection.
"""
# print(collection.peek(limit=10))


# ==================================================================
from pprint import pprint
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


collection = client.get_collection(
    name='netflix_titles',
    embedding_function=OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
)

result = collection.query(
    query_texts=["movies where people sing a lot"],
    n_results=3
)

pprint(result)


"""
will update the existing IDs if they exist.
"""
# collection.update(
#     ids=["id-1", "id-2"],
#     documents=["New document 1", "New document 2"]
# )

"""
will add the IDs if they don't exist.
"""
# collection.upsert(
#     ids=["id-1", "id-2"],
#     documents=["New document 1", "New document 2"]
# )

"""
will delete the documents with the specified IDs.
"""
# collection.delete(ids=["id-1"])

"""
will delete all documents from the collection.
"""
# client.reset()

