"""
Looking at how we can compute the similarity between two pieces of text using embeddings!
    - semantically similar texts are embedded more closely in the vector space
    - measuring distance allows us to measure similarity
    - enables embeddings applications:
        - semantic search
        - recommendation systems
        - classification

Measuring similarity:
    - cosine similarity is a common metric for measuring similarity between two vectors
"""
import pprint
from scipy.spatial import distance

distance = distance.cosine([0, 1], [1, 0])

# pprint.pprint(distance)

""" 
- Range from 0 to 2
- Smaller numbers = Greater similarity
"""

from common.helpers import create_embeddings, summarize_embeddings


articles = [
{"headline": "Economic Growth Continues Amid Global Uncertainty", "topic": "Business"},
{"headline": "Interest rates fall to historic lows", "topic": "Business"},
{"headline": "Scientists Make Breakthrough Discovery in Renewable Energy", "topic": "Science"},
{"headline": "India Successfully Lands Near Moon's South Pole", "topic": "Science"},
{"headline": "New Particle Discovered at CERN", "topic": "Science"},
{"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility", "topic": "Tech"},
{"headline": "Tech Giant Buys 49% Stake In AI Startup", "topic": "Tech"},
{"headline": "New Social Media Platform Has Everyone Talking!", "topic": "Tech"},
{"headline": "The Blues get promoted on the final day of the season!", "topic": "Sport"},
{"headline": "1.5 Billion Tune-in to the World Cup Final", "topic": "Sport"}
]

"""
create_embedding (now in common/helpers.py) is a custom function to send a request to the API, and extract and return embeddings from the response. This function can be called on a single string, or on a list of string and always returns a list of embeddings for the single string case, make sure to zero-index the function's result.
"""

# pprint.pprint(summarize_embeddings(create_embedding(["Python is the best!", "R is the best!"])), sort_dicts=False)

# pprint.pprint(summarize_embeddings(create_embedding("DataCamp is awesome!")))

"""
embed all headlines in one request and store each embedding on its article
"""
headline_embeddings = create_embeddings([article["headline"] for article in articles])
for article, embedding in zip(articles, headline_embeddings):
    article["embedding"] = embedding

"""
example:
    first we'll import distance from scipy.spatial for the cosine distance calculation, and NumPy to access its argmin function, which returns the index of the smallest value in a list.    
"""
from scipy.spatial import distance
import numpy as np

"""
Let's start with a piece of text to compare to our embedded headlines: computer
"""
search_text = "computer"

"""
we start by embedding this text using our create_embeddings custom function, remembering to zero-index the result
"""
search_embedding = create_embeddings(search_text)[0]

"""
to find the most similar headline to this text, we'll loop over each article, calculating the cosin distance between each embedded headline and the embedded query, we start by creating an empty list to store our distances.
"""
distances = []

"""
and loop over each article in our articles list of dictionaries.
"""
for article in articles:
    """
    next, we calculate the cosine distance between the text and headline by calling distance.cosine passing it the embedded text and deadline
    """
    dist = distance.cosine(search_embedding, article['embedding'])

    """
    finally we append this distance to the distances list, the most similar headline will have the smallest cosine distance, so we can use NumPy's argmin function to find the index of the smallest value in the distances list.
    """
    distances.append(dist)

"""
then use it to subset the article at this index and return its headline
"""
min_dist_ind = np.argmin(distances)

print(articles[min_dist_ind]['headline'])