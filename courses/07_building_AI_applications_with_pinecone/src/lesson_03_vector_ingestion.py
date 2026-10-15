
"""
Pinecone requires vectors to be structured in this particular way, so we may need to perform a bit of list and dictionary manipulation to get this point
"""
vectors = [
    {
        "id": "0",
        "values": [0.025525547564029694, 0.0188823901116848]
    },
    {
        "id": "9",
        "values": [0.020712468773126602, 0.006418442353606224]
    },
]

"""
Before we start ingesting these vectors, it's also important to check that they have the same dimensionality as the index.
To do this, we create a list comprehension that checks if the length of the list located under the 'values' key is 1536 for each vector in vectors. 
"""
vector_dims = [len(vector['values']) == 1536 for vector in vectors]

"""
To check that this condition is true for every vector, we can call the all() function on the list
"""
print(all(vector_dims))

"""
The .upsert() method is commonly used for ingesting vectors. Upsert is a combination of update and insert: it we try to ingest a vector ID that is already present in the index, it will get updated with the new data. If the vector isn't in the index, it will be inserted. We call the .upsert() method on our index, and pass it the vectors.
"""
# .upsert()
# index.upsert(
#     vectors=vectors
# )

"""
To check that the vectors were successfully upserted, we can call the .describe_index_stats() method. We can see all 10 vectors were inserted.
"""
# index.describe_index_stats()







