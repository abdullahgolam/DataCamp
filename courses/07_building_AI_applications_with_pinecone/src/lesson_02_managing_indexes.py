from pinecone import Pinecone, ServerlessSpec
from common.config import PINECONE_API_KEY
from pprint import pprint

pc = Pinecone(api_key=PINECONE_API_KEY)
# pc.create_index(
#     name="datacamp-index",
#     dimension=1536,
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1",
#     )
# )

"""
Connect to the index to begin ingesting vectors and querying them
"""

index = pc.Index(name="datacamp-index")

"""
Return information about the number of vectors in the index, the proportion of the index that is full - the fullness - and the dimensionality of the index, which should correspond to what was set when the index was created. There's also an empty dictionary to hold namespaces.
"""
pprint(index.describe_index_stats())

"""
Namespaces:
Namespaces are essentially containers for partitioning indexes into smaller units. This could be to keep distinct datasets or data versions separate, or perhaps to partition group in the parent dataset.
Queries and other manipulations are then limited to that namespace. For now, we'll focus on working within a single namespace, but later in the course we'll discuss distributing vectors and querying across multiple namespaces.
    In Summary:
        - Containers for partitioning indexes
            - Separate datasets
            - Data versioning
            - Separate groups
"""

"""
Organizations:
Organizations also play a key role in manageing indexes. They provide control over access and billing to different Pinecone projects. Pinecone projects can contain many indexes, so the Pinecone ecosystem consists of a hierarchy of organizations at the top, down to projects, indexes, namespaces, and individual records.
    types or org. roles:
        - org. owner: have permissions across the entire org.
            managing, billing, users, all projects.
        - org. user: restricted org.-level permissions, org owners invite them to the org. and assign them to specific project, of which, they become project owners.
"""

"""
Deleting indexes
"""
# pc.delete_index("datacamp-index")

pc.list_indexes()