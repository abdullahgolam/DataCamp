"""
Semantic search engines
    Semantic search engines use embeddings to return the most semantically similar results to a user input. This involves embedding and ingesting a series of documents to search through into a Pinecone index, embedding a user query, and using the user query vector to query the index and return results. We've seen how to ingest vectors and query indexes, so the final piece we need is to be able to embed the document and user query to create the vectors.
        
        1. Embed and ingest documents into a Pinecone index
        2. Embed a user query
        3. Query the index with the embedded user query

                                                                     * Pinecone
                                                                    .-------------.
        +--------------+                                           (               )
        |  Documents   |-------------+                             |`-------------'|
        +--------------+             |                             |               |
                                     v                             |               |
                          +--------------------+    +------------+ |               |
                          |     Embeddings     |   /            /  |    Vector     |
                          |       Model        |->/ Embeddings /-->|   Database    |
                          +--------------------+ /            /    |               |
                                     ^          +------------+     |               |
        +---------------+            |                             |               |
        || Application ||            |                             |               |
        ||  Interface  ||------------+                             |               |
        +---------------+   Query                                  (               )
               ^                                                    `-------------'
               |                        Result                             |
               +-----------------------------------------------------------+
"""

"""
Setting up Pinecone and OpenAI for semantic search
    For this, we'll use an OpenAI embedding model. We'll initialize Pinecone and OpenAI clients; and create a new serverless index specifically for our semantic search application. We give this index a dimensionality of 1536, which is the dimensionality of the vectors outputted by the OpenAI model we'll be using, if another embedding model is used, be sure to check the output dimensions and use that here. Finally, we connect to the index.
"""
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from common.config import OPENAI_API_KEY, PINECONE_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name='semantic-search-datacamp',
    dimension=1536,
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

index = pc.Index("semantic-search-datacamp")

"""
Ingesting documents to Pinecone index
    With our Pinecone index ready, let's prepare our documents for indexing. We'll use embedding and ingesting SQuAD dataset, comprising of Wikipedia articles that we can semantically search through the data with read_csv(), NumPy for a couple of data manipulation functions, and uuid, which we'll use to generate unique record IDs before upserting. The dataset includes columns for article id, text, and title.
"""
import pandas as pd
import numpy as np
from uuid import uuid4

df = pd.read_csv('../dataset/squad_dataset.csv')

"""
Ingesting documents to Pinecone
To begin ingesting documents into our Pinecone index, we set a batch_limit to process 100 documents per request for efficiency.
"""
batch_limit = 100

"""
We'll iterate through the documents in batches, using NumPy's array_split() function and batch_limit to create batches from the DataFrame. With each btch, we'll extract the metadata into a dictionary using a list comprehension. This creates a metadata into a dictionary using a list comprehension. This creates a metadata dictionary for each row in the batch, assigning the row values to dictionary keys. We also extract the article text to embed and create unique IDs for each row using the uuid4() function. 
"""
for batch in np.array_split(df, len(df) / batch_limit):
    metadatas = [{"text_id": row['id'], "text": row['text'], "title": row['title']} for _, row in batch.iterrows()]
    texts = batch['text'].tolist()
    ids = [str(uuid4()) for _ in range(len(texts))]

    """
    Next, we create a request to OpenAI's API to embed these texts, extracting the raw embeddings from the response.
    """
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeds = [np.array(x.embedding) for x in response.data]

    """
    Finally, we zip together the ids, vectors, and metadatas and upsert them to a nw namespace called squad_dataset.
    """
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace="squad_dataset")

"""
We can verify the upsertion with .describe_index_stats()
"""
index.describe_index_stats()
"""
{'dimension': 1536, 'index_fullness': 0.02,
'namespaces': {'squad_dataset': {'vector_count': 2000}},
"""

"""
Querying with Pinecone
Let's now perform semantic search with Pinecone! Consider the following query. To initiate semantic search, we encode this query sing the same embedding model again. It's crucial to align the embedding model used tor the query and documents, ensuring consistency in dimensions and processing methodology. Once the query is encoded, we query the namespace with it. We added the include_metadatas argument here so we can access the article text, which is stored in the metadata, for the returned records.
"""

query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
query_response = client.embeddings.create(
    input=query,
    model="text-embedding-3-small")
query_emb = query_response.data[0].embedding

retrieved_docs = index.query(vector=query_emb,
                             top_k=3,
                             namespace=namespace,
                             include_metadata=True)

"""
We can loop through the returned records and extract the similarity scores and text.
"""
for result in retrieved_docs['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")
"""
0.41: Architecturally, the school has a Catholic character. Atop the Main Building
gold dome is a golden statue of the Virgin Mary...
0.3: Because of its Catholic identity, a number of religious buildings stand on
campus. The Old College building has become one of two seminaries...
0.29: Within the white inescutcheon, the five quinas (small blue shields) with
their five white bezants representing the five wounds...
"""