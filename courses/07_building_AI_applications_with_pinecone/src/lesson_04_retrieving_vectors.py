"""
Accessing vectors:
    There are two main methods for accessing vectors from a pinecone vector database: Fetching and querying.
    Fetching:
        involves finding and returning a record using its ID. This is normally done to explore and verify particular records from the index.
    Querying:
        involves sending a vector to the index, and returning the vector or vectors that are most similar to it.

    if vectors were books, fetching would be like using the ISBN, or international Standard Book Number, to find a book. Querying is like searching for books that are similar to one we already own.
"""

"""
Fetching vectors
    To fetch records based on their IDs, we can call the .fetch() method, passing it a list of record IDs to return. 
"""
# index.fetch(
#     ids=['0', '1']
# )
"""
{'namespace': '',
'usage': {'read_units': 1},
'vectors': {'0': {'id': '0',
                    'metadata': {"genre": "productivity", "year": 2020},
                    'values': [0.025525547564029694, ...]},
            '1': {'id': '1',
                    'metadata': {"genre": "action", "year": 2023},
                    'values': [-0.0131468913, ...]}}
}

We can see the result contains a dictionary under the 'vectors' key that contains one entry for each record returned. From these entries, we can access the vector values and associated metadata. There's also a 'usage' key, which shows we've used one read unit when fetching our vectors.
"""

"""
Read units:
    Read units, or RUs, are a Pinecone specific measure of the resources consumed during reading operations. Fetching, querying, and listing vectors all consume read units. For fetching, 1RU is consumed for every batch of 10 records retrieved. Fortunately, Pinecone's starter account provides a generous monthly quota of read units for developing and testing solutions before scaling.
"""

"""
Fetching vectors from namespaces
    We can also fetch vectors from specific namespaces by specifying the namespace argument
"""
# index.fetch(
#     ids=['0', '1']
#     namespace='namespace1'
# )
"""
{'namespace': 'namespace1',
'usage': {'read_units': 1},
'vectors': {'0': {'id': '0',
                    'metadata': {"genre": "productivity", "year": 2020},
                    'values': [0.025525547564029694, ...]},
            '1': {'id': '1',
                    'metadata': {"genre": "action", "year": 2023},
                    'values': [-0.0131468913, ...]}}
}

We can see that the previously blank namespace output has now been populated.
"""