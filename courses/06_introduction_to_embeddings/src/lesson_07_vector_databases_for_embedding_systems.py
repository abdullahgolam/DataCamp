"""
Limitations of the current approach:
    - Loading all the embeddings into memory (1536 floats ~ 13kB/embedding)
    - Recalculated embeddings for each new query
    - Calculating cosine distances for every embedding and sorting is slow and scales Linearly

Vector databases:
    - Embedded documents are stored and queried from the vector database

NoSQL databases:
    - More flexible structure that allows for faster querying
    - MongoDB, DynamoDB, CouchDB, etc.

SQL/Relational databases:
    - Structured data into tables, rows, and columns
    - PostgreSQL, MySQL, SQLite, etc.

Components to store:
    - Embeddings
    - Source texts
    - Metadata
        - IDs and references
        - Additional data useful for filtering results
    Top tip: Don't store the source text as metadata!

Which solution is best?
    - Database management:
        - Managed -> more expensive but lowers workload
        - Self-managed -> cheaper but requires time and expertise
    - Open source or commercial?
        - Open source -> flexible and cost-effective
        - Commercial -> better support, more advanced features, and compliance
    - Data models: does the type of data lend itself to a particular database type?
    - Specific features: does your use case depend on specific functionality, such as multi-modal storage?

In this course we'll be using Chroma, as it's open-source and quick to set up.
"""

