"""
RAG chatbot with Pinecone and OpenAI
    Retrieval Augmented Generation (RAG) combines a vector database with an LLM. The user query is used to retrieve the most similar documents from the vector database, which are then combined with the query into a single prompt. The LLM answers using this extra context, and the response is returned to the application.

        1. Query the vector database with the user query
        2. Retrieve the most similar documents
        3. Build a prompt from the user query + similar documents
        4. Send the prompt to the LLM
        5. Return the model response to the application

                         .-------------.
                        (               )
                        |`-------------'|
                        |    Vector     |     +-----------------+
                        |   Database    |---->|  Most Similar   |
                        |               |     |   Documents     |
                        `---------------'     +-----------------+
                                ^                      |
                                |                      |
           +--------------------+                      |
           |                                           v
    +--------------+      User Query        +------------------+          +-----------+
    | Application  |----------------------->/ Prompt =         /--------->|           |
    |  Interface   |                       / User Query +     /           |    LLM    |
    +--------------+                      / Similar Docs     /            |           |
           ^                             +------------------+             +-----------+
           |                                                                    |
           +--------------------------------------------------------------------+
                                        Model Response

Retrieval Augmented Generation (RAG):
    1. Embed user query
    2. Retrieve similar documents
    3. Added documents to prompt
"""

"""
We'll implement a question-answering chatbot using Pinecone and OpenAI.
"""
from openai import OpenAI
from pinecone import Pinecone
import pandas as pd
from uuid import uuid4
from common.config import OPENAI_API_KEY, PINECONE_API_KEY


client = OpenAI(api_key="OPENAI_API_KEY")
pc = Pinecone(api_key="PINECONE_API_KEY")

index = pc.Index("semantic-search-datacamp")

youtube_df = pd.read_csv("data/youtube_reg_data.csv")

"""
| id | blob | channel_id | end | published | start | text | title | url |
|----|------|------------|-----------------|-------|------|-------|-----|
|int | dict | str        | int | datetime  | int   | str  | str   | str |
"""

"""
We'll upsert this dataset in batch sizes of 100
"""
batch_limit = 100

"""
Split the dataset using NumPy's array_split() function. With each batch, we'll extract the metadata using a list comprehension. 
"""
for batch in np.array_split(youtube_df, len(youtube_df) / batch_limit):
    """
    This creates a metadata dictionary for each row in the batch, assigning the row values to dictionary keys.
    """
    metadatas = [{"text_id": row['id'], "text": row['text'],
                  "title": row['title'], "url": row['url'],
                  "published": row['published']} for _, row in batch.iterrows()]

    """
    We also extract the article text to embed and create unique IDs for each row using the uuid4() function.
    """
    texts = batch['text'].tolist()
    ids = [str(uuid4()) for _ in range(len(texts))]

    """
    Next we create a request to OpenAI to embed these texts, extracting the raw embeddings from the response.
    """
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeds = [np.array(x.embedding) for x in response.data]

    """
    Finally, we zip together the ids, vectors, and metadatas and upsert them to the youtube_rag_dataset namespace.
    """
    index.upsert(vectors=zip(ids, embeds, metadatas), namespace='youtube_rag_dataset')

"""
Now that our documents are ingested, let's implement a function to retrieve relevant documents from a Pinecone index based on a query.
"""
def retrieve(query, top_k, namespace, emb_model):
    """
    It uses the same OpenAI embeddings function to encode the user query, extracting the embeddings from the API response.
    """
    query_response = client.embeddings.create(input=query, model=emb_model)
    query_emb = query_response.data[0].embedding

    """
    The query embeddings are used to query our 'youtube_rag_dataset' namespace to retrieve the most relevant documents.
    """
    retrieved_docs = []
    sources = []
    docs = index.query(vector=query_emb, top_k=top_k,
                       namespace='youtube_rag_dataset', include_metadata=True)

    """
    For each retrieved document, we extract the text and source metadata, appending them to the retrieved_docs and sources list, respectively.
    """
    for doc in docs['matches']:
        retrieved_docs.append(doc['metadata']['text'])
        sources.append((doc['metadata']['title'], doc['metadata']['url']))

    """
    Finally, it returns the lists of retrieved documents and their corresponding sources.
    """

"""
Here's the function in action, retrieving the transcripts based on the user's query along with the video's title and URL.
"""
query = "How to build next-level Q&A with OpenAI"
documents, sources = retrieve(query, top_k=3, namespace='youtube_rag_dataset', emb_model="text-embedding-3-small")
"""
Document: To use for Open Domain Question Answering. We're going to start...
Source: How to build a Q&A AI in Python [...], https://youtu.be/w1dMEWm7jBc

Document: Over here we have Google and we can ask Google questions...
Source: How to build next-level Q&A with OpenAI, https://youtu.be/coaaSxys5so

Document: We need vector database to enhance the quality of Q&A systems...
Source: How to Build Custom Q&A Transfo [...], https://youtu.be/ZIRmXKHp0-c
"""

"""
Now that we have the top_k documents, we'll construct a prompt_with_context_builder() function to create a contextual prompt for our model. It glues together and instruction in prompt_start, the most relevant documents, which are joined together with the .join() method, and the user query in prompt_end.
"""
def prompt_with_context_builder(query, docs):
    delim = '\n\n---\n\n'
    prompt_start = 'Answer the question based on the context below.\n\nContext:\n'
    prompt_end = f'\n\nQuestion: {query}\nAnswer:'
    prompt = prompt_start + delim.join(docs) + prompt_end
    return prompt

"""
Here's prompt_with_context_builder() in action. The final prompt includes the retrieved document followed by the user's query. This prompt will be used to generate the model response.
"""
query = "How to build next-level Q&A with OpenAI"
context_prompt = prompt_with_context_builder(query, documents)
"""
Answer the question based on the context below.

Context:
to use for Open Domain Question Answering. We're going to start with...
Over here we have Google and we can ask Google questions...
We need vector database to enhance the quality of Q&A systems...

Question: How to build next-level Q&A with OpenAI
Answer:
"""

"""
Finally, let's create a question_answering() function. I takes a prompt string, document sources, and uses the OpenAI chat completion endpoint.
"""
def question_answering(prompt, sources, chat_model):
    """
    We'll define two prompts: system prompt, setting the model's behavior, and a user prompt, which will be the output from our prompt_with_context_builder() function. The function returns the final answer with document sources.
    """
    sys_prompt = "You are a helpful assistant that always answers questions."
    res = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": prompt}
                  ], temperature=0)
    answer = res.choices[0].message.content.strip()
    answer += "\n\nSources:"
    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    return answer

"""
Here's the function in action. Note there are three documents, or transcript segments, based on the user's query.
"""
query = "How to build next-level Q&A with OpenAI"
answer = question_answering(prompt_with_context, sources, chat_model='gpt-4o-mini')
"""
To build next-level Q&A with OpenAI, we need vector database to enhance the quality of Q&A systems...

Sources:
How to build a Q&A AI in Python [...], https://youtu.be/w1dMEWm7jBc
How to build next-level Q&A with OpenAI, https://youtu.be/coaaSxys5so
How to Build Custom Q&A Transfo [...], https://youtu.be/ZIRmXKHp0-c
"""

"""
Putting it all together, here's how all the functions work together from user query to the final generated answer.
"""
query = "How to build next-level Q&A with OpenAI"
documents, sources = retrieve(query, top_k=3, namespace='youtube_rag_dataset', emb_model="text-embedding-3-small")
prompt_with_context = prompt_with_context_builder(query, documents)
answer = question_answering(prompt_with_context, sources,chat_model='gpt-4o-mini')
























