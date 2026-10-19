from common.config import PINECONE_API_KEY
from pinecone import Pinecone


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(name="datacamp-index")

"""
Multitenancy:
    is a software architecture where a system can serve multipl groups of users(more generally referred to as tenants)in isolation.
    In Pinecone, this typically refers to storing distinct datasets in separate namespaces while being served by a single index.
    This architecture not only allows us to separate data from different groups of users, to ensure security and privacy, but can also reduce query latency, as queries can be directed to specific namespaces, reducing the search space.

    - Serve multiple tenants in isolation
    - Separate different customers' data
        - Security and privacy
    - Reduce query latency

Multitenancy strategies:
    1. Namespaces
        - Advantages;Reduces the need for additional indexes
        - Disadvantages: Tenants share resources, complex data
    2. Metadata Filtering
        - Advantages; Allows querying across multiple tenants
        - Disadvantages: Shared resources, challenging cost tracking
    3. Separate Indexes
        - Advantages; Physically separates tenants, allocates individual resources
        - Disadvantages: Requires more effort and cost
"""

"""
Namespaces
    - Created implicitly during upsertion if they don't exist
    
    For example, if we upsert vector_set1 into namespace1 using the namespace argument, namespace1 will be automatically created if it doesn't already exist. We can continue to create more namespaces by specifying new namespace names when upserting the different sets of vectors.
"""
vector = []
vector_set1 = []
vector_set2 = []

index.upsert(
    vectors=vector_set1, namespace="namespace1",
)
index.upsert(
    vectors=vector_set2, namespace="namespace2",
)

"""
We can inspect our namespaces with the .describe_index_stats() method we've used previously.
"""
index.describe_index_stats()

"""
{'dimension': 1536,
'index_fullness': 0.0,
'namespaces': {'namespace1': {'vector_count': 5},
                'namespace2': {'vector_count': 5}},
'total_vector_count': 10}

In the output, we can see that the vectors have been successfully upserted and distributed into the two new namespaces.
"""

"""
Querying vectors from namespaces
    This is nly requires one additional argument to indicate the namespace name that should be queried. Querying isn't the only operation that can be done on the namespace level; in fact, all of index operations we've looked at can direted to specific namespaces.
"""
query_result = index.query(
    vector=vector,
    namespace='namespace1',
    top_k=3
)

"""
We can delete vector IDs from specific namespaces. This is particularly useful, as IDs only need to be unique within a namespace, so there may be other records in different namespaces with the same ID, and we wouldn't want to delete those accidentally.
"""
index.delete(
    ids=["1", "2"],
    namespace='namespace1'
)

