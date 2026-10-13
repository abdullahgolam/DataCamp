"""
Querying vectors:
    Querying: receive the most semantically similar vectors to an input vector.

    querying is used in all kinds of AI apps. In chatbots, user prompts can be embedded and used to query a vector database and provide the model used with more relevant information. In effect, it gives the model more context than it had from its training data. Similarly in semantic search engines, user searches are embedded and used to query the database to return the most relevant results.

"""

"""
We can use the .query() method to query the index. We need to pass it a vector to query it with, and the top_k argument, which is the number of results to return the top_k most similar vectors. As with other index methods, the query vector will need to have the same dimensionality as the index and other vectors in the index.
"""
index.query(
    vector=[-0.25919762305275, ...]
)

"""
{'matches': [{'id': '1', 'score': 0.0478537641, 'values': []},
             {'id': '2', 'score': 0.046000585, 'values': []},
             {'id': '3', 'score': 0.0458319113, 'values': []}],
'namespace': '',
'usage': {'read_units': 5}}

The output shows te IDs of the records with the most similar vectors. Each record is also given a score, which is a measure of the vector's similarity to the input vector. The vector values also aren't shown by default for brevity.
"""

"""
To return the vector values in the output, set the include_values argument to True. 
"""
index.query(
    vector=[-0.250919762305275, ...],
    top_k=3,
    include_values=True
)

"""
{'matches': [{'id': '1', 'score': 0.0478537641, 'values': [-0.0131468913, ...]},
             {'id': '2', 'score': 0.046000585, 'values': [-0.0120476764, ...]},
             {'id': '3', 'score': 0.0458319113, 'values': [0.00285418332, ...]}],
'namespace': '',
'usage': {'read_units': 5}}

Notice for both of these queries, we consumed five read units.
"""

"""
Read units (RUs) for querying
    Querying, like fetching, also consumes read units. For querying, calculating the number of read units that will be consumed in an operation is less straightforward than for fetching records. The number of RUs is dependent on the number of records in the namespace being read and the size of the records, which is itself dependent on the dimensionality of the vectors and the amount of metadata stored. Pinecone provide a guide on RUs consumed for different configurations, but if these numbers don't reflect your project, you can dive deeper into Pinecone's documentation. 
    
    - For querying, RUs is harder to calculate
    - Dependent on:
        - No. of records in the namespace
        - Size of records
            Vector dimensionality
            Amount of metadata 
            
    Records per namespace           Dimension=384           Dimension=768           Dimension=1536
    1000,000                        5 RUs                   5 RUs                   6 RUs
    1,000,000                       16 RUs                  10 RUs                  18 RUs
    10,000,000                      18 RUs                  32 RUs                  59 RUs
"""

"""
Distance metrics
    So how does the query method actually determine the most similar vectors?
    What happens is we compute the distance between the query vector and vectors in the index, and the vectors with the smallest distance are returned. There are different metrics for calculating the distance between vectors, with the cosine similarity being one of the most common and the default in Pinecone.
    
    There's also the Euclidean distance, which is essentially a straight-line distance, and the dot product.
"""

"""
Setting the distance metric
    Determining the most appropriate distance metric can require some experimentation. The distance metric is set when creating the index using the metic argument, and can't be changed afterwards. The argument can take the strings 'cosine', euclidean', or 'dotproduct'
"""
pc.create_index(
    name="datacamp-index",
    dimension=1536,
    metric='dotproduct',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)