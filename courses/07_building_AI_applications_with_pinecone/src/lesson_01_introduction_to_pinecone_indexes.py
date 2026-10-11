"""
Pinecone: is a leading fully-managed vector database solution for building and scaling generative AI applications.
    it uses embeddings to store and retrieve data based on their semantic similarity.
        chatbots
        search engines
        recommendation engines

The output of this course is to learn to:
    Create indexes and ingest vectors
    Retrieve and query vectors
    Create common AI applications

Indexes:
    Indexes are one of the core components of the Pinecone Infrastructure. They are used to store vectors and serve queries and other manipulations over the vectors it contains.
    We can think of indexes like draws in a filing cabinet; each draw contains folders of information, and indexes contain records. Each vector is contained in a record, and the record also stores additional metadata that can be used in querying.
    It's also possible to create multiple indexes to house different datasets, like the different draws in the cabinet.
    In summary:
        - Store vectors
        - Serve queries and other vector manipulations
        - Index contains records for each vector, including metadata
        - Can create multiple indexes

Serverless vs. pod-based indexes
There are two types of indexes in pinecone: serverless and pod-based indexes.
In pod-based indexes, we choose one or more pre-configured units of hardware, called pods, in which to create the index, Depending on the pod type chosen, we get different amounts of storage, query latency, and query throughput. Serverless indexes don't required managing resources; instead, they scale automatically depending on usage. Serverless indexes run as a managed service on a cloud platform, like AWS, and the vectors are stored in blob storage on that platform. In many cases, serverless indexes result in lower costs than pod-based indexes, which is leading many organizations to switch to serverless. In this course, we'll be using serverless indexes, as these are accessible with pinecone's rree starter plan. However, much of what we'll learn will also apply to pod-based indexes.
    In Summary:
        Pod-Based
            - Choose hardware to create the index -> pods
            - Pod type determines storage, query latency, query throughout

        Serverless
            - No resource management
            - Indexes scale automatically
            - Run on cloud and store in blob
            - Easier to user and often lower cost
"""