"""

In this lesson, you'll learn how to build a question-answering chatbot using Pinecone and OpenAI, focusing on Retrieval-Augmented Generation (RAG) to enhance model responses.

Key points include:

    - Embedding user queries to retrieve relevant documents
    - Using metadata and unique IDs for dataset management
    - Creating contextual prompts for improved model responses

Here's a snippet of the process:
"""
def retrieve_documents(query):
    query_embedding = openai_embed(query)
    results = pinecone_index.query(query_embedding)
    retrieved_docs = [doc['text'] for doc in results]
    return retrieved_docs

def prompt_with_context_builder(query, top_docs):
    context = "\n".join(top_docs)
    return f"Context: {context}\nQuery: {query}"

def question_answering(prompt):
    response = openai_chat_completion(prompt)
    return response


