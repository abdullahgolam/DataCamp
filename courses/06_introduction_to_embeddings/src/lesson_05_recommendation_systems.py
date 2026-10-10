"""
Recommendation systems with embeddings
    - very similar to semantic search!
    - Process:
        1. Embed the potential recommendations and data point
        2. Calculate cosine distances
        3. Recommend closest items
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
     "keywords": ["technology", "investment", "artificial intelligence"]},
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

current_article = {
    "headline": "How NVIDIA GUPs Could Decide Who Wins the AI Race",
    "topic": "Tech",
    "keywords": ["ai", "business", "computers"]
}

def create_article_text(article):
    return f"""
    Headline: {article['headline']}
    Topic: {article['topic']}
    Keywords: {', '.join(article['keywords'])}
    """

article_texts = [create_article_text(article) for article in articles]
current_article_text = create_article_text(current_article)
# print(current_article_text)

from common.helpers import create_embeddings
from scipy.spatial.distance import cosine
import numpy as np


current_article_embeddings = create_embeddings(current_article_text)[0]
article_embeddings = create_embeddings(article_texts)

def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[0:n]

hits = find_n_closest(current_article_embeddings, article_embeddings)

for hit in hits:
    article = articles[hit['index']]
    # print(article['headline'])

user_history = [
    {"headline": "How NVIDIA GUPs Could Decide Who Wins the AI Race",
     "topic": "Tech",
     "keywords": ["ai", "business", "computers"]},
    {"headline": "Tech Giant Buys 49% Stake In AI Startup",
     "topic": "Tech",
     "keywords": ["business", "AI"]}
]

"""
Recommendations on multiple data points
Process:
    - Combine multiple vectors into one by taking the mean
    - Compute cosine distances
    - Recommend closest vector
        - Ensure that it's unread 
"""

history_texts = [create_article_text(article) for article in user_history]
history_embeddings = create_embeddings(history_texts)

"""
We take the mean to aggregate the two vectors into one that we can compare with the other articles.
"""
mean_history_embedding = np.mean(history_embeddings, axis=0)

"""
For the articles to recommend, we filter the list so it only contains articles not in the user_history.
"""
articles_filtered = [article for article in articles if article not in user_history]

"""
We combine the features and embed the text.
"""
article_texts = [create_article_text(article) for article in articles_filtered]
article_embeddings = create_embeddings(article_texts)

hits = find_n_closest(mean_history_embedding, article_embeddings)

for hit in hits:
    article = articles_filtered[hit['index']]
    print(article['headline'])