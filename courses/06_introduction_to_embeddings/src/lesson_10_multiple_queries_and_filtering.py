"""
Movie recommendations based on multiple datapoints
    We'll recommend movies related to other titles that a user has seen. Let's assume a user has seen: Terrifier (id: "s8170"), which is a horror film, and Strawberry Shortcake: Berry Bitty Adventures (id: "s8103", a kid's TV show. It's an odd combination, but hopefully it will help differentiate the recommendations.
"""
import chromadb
from pprint import pprint
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from common.config import OPENAI_API_KEY


client = chromadb.PersistentClient(path=r'D:\chromadb')

reference_ids = ["s8", "s9"]

collection = client.get_collection(
    name='netflix_titles',
    # embedding_function=OpenAIEmbeddingFunction(
    #     model_name="text-embedding-3-small",
    #     api_key=OPENAI_API_KEY
    # )
)

reference_texts = collection.get(ids=reference_ids)["documents"]

# result = collection.query(
#     query_texts=reference_texts,
#     n_results=3
# )

# pprint(result)

# get_ten = collection.peek(limit=10)

# pprint(list(zip(get_ten["ids"], get_ten["documents"])))

# pprint(reference_texts)

"""
This code to load the CSV file by creating a list to store the metadata, and populating it with the type and release_year from each row of the file, stored together in a dictionary.we also create a list of IDs so we can add the metadata to the existing items.
"""
import csv
# from pathlib import Path
# DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "netflix_titles_1000.csv"

# ids = []
# metadatas = []

# with open(DATA_PATH, encoding='utf-8', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for i, row in enumerate(reader):
#         ids.append(row['show_id'])
#         metadatas.append({
#             "type": row['type'],
#             "release_year": int(row['release_year'])
#         })

# collection.update(ids=ids, metadatas=metadatas)

result = collection.query(
    query_texts=reference_texts,
    n_results=3,
    # where={
    #     "type": "Movie"
    # }
    where={
        "$and": [
            {"type":
                {"$eq": "Movie"}
            },
            {"release_year":
                {"$gt": 2020}
            }
        ]
    }
)

"""
Where operators
where={
    "type": "Movie"
}

where={
    "type": {
        "$eq": "Movie"
    }    
}

List of operators: https://docs.chroma.ai/querying#where-operators
$eq - equal to, 
$ne - not equal to,
$gt - greater than, 
$gte - greater than or equal to, 
$lt - less than, 
$lte - less than or equal to, 
$in - in a list, 
$nin - not in a list, 
$contains - substring, 
$startswith - starts with, 
$endswith - ends with

Multiple where filters
Where={
    "$and": [
        {"type":
            {"$eq": "Movie"} 
        },
        {"release_year": 
            {"$gt": 2020}
        }
    ]
}

$or : filter based on at least one condition
"""

pprint(result)