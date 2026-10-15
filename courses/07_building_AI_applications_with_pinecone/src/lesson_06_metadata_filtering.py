"""
Metadata filtering:
    Vectors often have associated metadata stored in the index. This metadata can be strings, numbers, Booleans, and lists of strings. This metadata can be used to filter the vectors that are searched through during the query, reducing the search space and query latency.
        - Metadata can be strings, numbers, Booleans, and lists of strings
        - Metadata filtering: reduces search space and query latency

    {
        "genre": "action",
        "year": 2020,
        "color": "blue",
        "fit": "straight",
        "price": 29.99,
        "is_jeans": true,
        "areas": ["London", "Kent", "Bath"]
    }
"""

"""
To perform metadata filtering, we need to use the filter parameter n the .query() method. This argument takes a dictionary where each key is the metadata to filter, and their values specify how to filter them.
"""
index.query(
    vector=[-0.250919762305275, ...],
    filter={
    "genre": {"$eq": "documentary"},
    "year": 2019
    },
    top_k=1,
    include_metadatas=True
)
"""
In this case, 'year' is set to 2019 will be filtered and queried. There's also a second metadata filter, where 'genre' is set to another dictionary where they key is the $eq operator, so this filter selects records with a 'genre' equal to 'documentary'. The two filters here show two methods of performing an equality filter, so you might be wondering why the dictionary and operator version is useful.
include_metadatas argument to include the metadata in the result, so we can verify that the filter worked.
"""
















