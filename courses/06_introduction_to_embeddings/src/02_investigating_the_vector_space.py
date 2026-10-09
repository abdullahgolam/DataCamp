from openai import OpenAI
from common.config import OPENAI_API_KEY
from common.helpers import summarize_embeddings
from pprint import pprint

client = OpenAI(api_key=OPENAI_API_KEY)

""" a list of dictionaries, each containing a headline and its corresponding topic"""
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

headline_text = [article["headline"] for article in articles]
# print(headline_text)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=headline_text
)

response_dict = response.model_dump()
# pprint(response_dict)

""" to extract these embeddings from the response and store them in the articles list of dictionaries, 
we loop over the indexes and articles using enumerate. For each article, we assign the embedding at the same index in the response to the article's embedding key."""
for i, article in enumerate(articles):
    article['embedding'] =response_dict['data'][i]['embedding']

# pprint(summarize_embeddings(articles[:2]), sort_dicts=False)

"""prints the length of the embedding vector for the first article"""
# pprint(len(articles[0]['embedding']))
""" the embedding model returned 1536 number repreenting the semantic meaning of its headline, or in other words, its position, or vector, in the vector space."""

""" let's take a look at another, longer headline."""
# pprint(len(articles[5]['embedding']))
""" we get 1536 again, this is a key property of OpenAI's embedding models - they always return 1536 numbers, no matter the input """


"""
Dimensionality reduction and t-SNE
    - various techniques to reduce the number of dimensions
    - t-SNE (t-distributed Stochastic Neighbor Embedding)
    
    t-SNE can be implemented using the scikit-learn, a popular Python package for machine learning tasks.
"""

from sklearn.manifold import TSNE
import numpy as np


embeddings = [article['embedding'] for article in articles]

tsne = TSNE(n_components=2, perplexity=5)
"""
n_components: The number of dimensions to reduce the data to. In this case, we want to reduce the embeddings to 2 dimensions for visualization purposes.
perplexity: A hyperparameter that controls the balance between local and global aspects of the data
"""

embeddings_2d = tsne.fit_transform(np.array(embeddings))

"""
The fit_transform method of the TSNE object is called with the embeddings as input. This method performs the dimensionality reduction and returns the transformed embeddings in 2D space.
"""

import matplotlib.pyplot as plt
"""
to visualize these transformed embeddings, we call plt-dot-scatter from Matplotlib
"""

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
"""
on the first and second columns of the embeddings_2d array, which represent the x and y coordinates of the points in the 2D space.
"""

topics = [article['topic'] for article in articles]
for i, topic in enumerate(topics):
    plt.annotate(topic, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()
"""
this code snippet to extract the article topics, annotate the plot with them, and display the plot
"""

"""
Visualizing the embeddings
    - similar articles are grouped together!
    - model captured the semantic meaning
"""