"""
Semantic search
    - Use embeddings to return the most similar results to a search query
    - Example: Semantic search for online news website

There are three steps to semantic search:
    - embed the search query and texts to compare against
    - compute the cosine distances between the embedded search query and other embedded texts
    - extract the texts with the smallest cosine distance
"""

articles = [
    {"headline": "Economic Growth Continues Amid Global Uncertainty",
     "topic": "Business",
     "keywords": ["economy", "business", "finance"]},
    {"headline": "Interest rates fall to historic lows",
     "topic": "Business",
     "keywords": ["economy", "business", "finance"]},
    {"headline": "Scientists Make Breakthrough Discovery in Renewable Energy",
     "topic": "Science",
     "keywords": ["science", "innovation", "renewable"]},
    {"headline": "India Successfully Lands Near Moon's South Pole",
     "topic": "Science",
     "keywords": ["space", "exploration", "india", "AI"]},
    {"headline": "New Particle Discovered at CERN",
     "topic": "Science",
     "keywords": ["physics", "research", "discovery"]},
    {"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility",
     "topic": "Tech",
     "keywords": ["technology", "innovation", "accessibility"]},
    {"headline": "Tech Giant Buys 49% Stake In AI Startup",
     "topic": "Tech",
     "keywords": ["technology", "investment", "artificial intelligence"]},
    {"headline": "New Social Media Platform Has Everyone Talking!",
     "topic": "Tech",
     "keywords": ["technology", "social media", "platform"]},
    {"headline": "The Blues get promoted on the final day of the season!",
     "topic": "Sport",
     "keywords": ["football", "soccer", "promotion"]},
    {"headline": "1.5 Billion Tune-in to the World Cup Final",
     "topic": "Sport",
     "keywords": ["football", "soccer", "world cup"]}
]

from pprint import pprint
from common.helpers import create_embeddings, summarize_embeddings

"""
we'll define a function called create_article_text, this function uses an F-string, or formatted string, to return the desired string structure. f-string allow us to insert variables into strings without having to convert them into strings and concatenate them. F-strings are created by specifying on f before the quotes, and note that we've defined a multi-line string using triple quotes. To insert an object, we use curly brackets and specify the variable or other Python code to insert. For the article headline and topic, these values are extracted using their keys and inserted into the string at the desired locations. The keywords are a little trickier because they were stored as a list rather than a string. To convert the keywords list into a string, we use the join list method, which joins the contents of the list together into a single string. The method is called on the string we want to delimit each keyword with, in this case, a comma and space. 
"""
def create_article_text(article):
    return f"""
    Headline: {article['headline']}
    Topic: {article['topic']}
    Keywords: {', '.join(article['keywords'])}
    """

"""
Calling the function on the final headline shows the text in the desired formatted string.
"""
# print(create_article_text(articles[-1]))

"""
To apply the function and combine the features for each article, we use a list comprehension, calling our function on each article in articles.
"""
article_texts = [create_article_text(article) for article in articles]

"""
Finally, to embed these strings, we call the create_embedding function on the result. 
"""
article_embeddings = create_embeddings(article_texts)

"""
Recall, that this creates a list of embeddings for each input using the OpenAI API.
"""
# pprint(summarize_embeddings(article_embeddings), sort_dicts=False)

"""
Now that we have our embeddings, it's time to compute cosine distances
"""

from scipy.spatial.distance import cosine

"""
We'll define a function called find_n_closest, that takes a query_vector, the embedded search query, and embeddings to compare against, our embedded articles, and returns the n most similar results based on their cosine distances.
"""
def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    """
    for each embedding, we calculate the cosine distance to the query_vector, and store it in a dictionary along with the embedding's index, which we append to a list called distances.
    """
    for index, embedding in enumerate(embeddings):
        dist = cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    """
    To sort the distances list by the distance key in each dictionary, we use the sorted function and its key argument. The key argument takes a function to evaluate each dictionary in distances and sort by; in this case, it's a lambda function that accesses the distance kdy from each dictionary.
    """
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    """
    Finally, the function returns the closest n results.
    """
    return distances_sorted[0:n]

"""
Time to bring all the semantic search pleces together! We'll query our embeddings using the text, "AI".
"""
query_text = "AI"
"""
First, we embed the search query using our create_embeddings function and extract its embeddings by zero-indexing.
"""
query_vector = create_embeddings(query_text)[0]
"""
Next, we use the find_n_closest function to find the three closest hits based on our article_embeddings.
"""
hits = find_n_closest(query_vector, article_embeddings)
"""
Finally, to extract the most similar headlines, we loop through each hit, using the hit's index to subset the corresponding headline, and print.
"""
for hit in hits:
    article = articles[hit['index']]
    print(article['headline'])
    """
    As we'd expect, the top result specifically mentions AI, and the others are on similar topics.
    """
