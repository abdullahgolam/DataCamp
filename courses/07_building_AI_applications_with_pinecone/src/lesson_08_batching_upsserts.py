"""
Upserting limitations
    1. Rate of requests
    2. Size of requests

    These limits differ depending on the account type, but for som use cases, strategies have to be implemented to work around these limitations.

Batching: breaking requests up into smaller chunks
"""

"""
To start, we'll create a chunks() function that will break our long list of vectors into smaller chunks containing batch_size vectors, 100, by default. The first step is to convert the list, which is an iterable, into on iterator with the iter() function. Recall that iterables are data types like strings, lists, and dictionaries where elements can be extracted one at a time. Iterators can be extracted one at a time. Iterators only produce elements on-demand, and return data in streams, which we need for our case. Next we create the first chunk by calling itertools.islice() on the iterator and specifying the batch_size parameter. We also wrap it in a tuple. We use a while loop that continues until there are no chunks left, yield the current chunk, which can then be upserted, and then update the cunk with the next batch of vectors using the same code that we used to define the first chunk.
"""
def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

"""
Sequential batching
    - Splitting requests and sending them sequentially one-by-one
    
We initialize our client and connect to the index. Then, we use our chunks() function to create on iterator and iterate over it, upserting each chunk of vectors until every chunk has been upserted.
"""
pc.Pinecone(api_key="YOUR API KEY")
index = pc.Index('datacamp-index')

for chunk in chunks(vectors):
    index.upsert(vectors=chunk)
"""
This solves the problem of rate and size limiting, but does come at the cost of speed - this can be really slow! To speed up batch upsertion, parallelizing requests is usually the best option.
"""

"""
In parallel batching, many batches can sent at the same time, or in parallel. To enable parallel request, initialize the pinecone client with the pool_threads parameter, which sets the maximum number of simultaneous requests.
"""
pc = Pinecone(api_key="YOUR_API_KEY", pool_threads=30)

"""
Next, we start a with statement to connect to the index, again, specifying the pool_threads parameter. Next, we begin the upsertion, which we do asynchronously so that requests can be sent independently. This is indicated with the async_req parameter, which is set to True in this case. The rest of the code is a list comprehension that upserts each chunk created by our chunks() function and assigns the asynchronous results to an object. Finally, we call the .get() method on each asynchronous result to wait for and retrieve the responses. Although there's quite a bit of extra code here, batching in parallel can provide huge speed boosts.
"""
with pc.Index('datacamp-index', pool_threads=30) as index:
    async_results = [index.upsert(vectors=chunk, async_req=True)
        for chunk in chunks(vectors, batch_size=100)]

    [async_result.get() for async_result in async_results]