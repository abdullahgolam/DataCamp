"""
Classification tasks
    Can take different forms, but generally, they involve assigning labels to items.

    Common tasks include
        - Assigning labels to items
            - Categorization
                - Example: headlines into topics
            - Sentiment analysis
                - Example: Classifying reviews as positive or negative
    Embeddings capture semantic meaning

Classification with embeddings
    - Zero-shot classification
        - Not using labeled data

    Process:
        1. Embed class descriptions
        2. Embed the item to classify
        3. Calculate cosine distances
        4. Assign the most similar label
"""
from common.helpers import create_embeddings
from scipy.spatial.distance import cosine

"""
The topic classes we'll be categorizing with.
"""
topics = [
    {'label': 'Tech', 'description': 'A news article about technology'},
    {'label': 'Science', 'description': 'A news article about science'},
    {'label': 'Health', 'description': 'A news article about health'},
    {'label': 'Entertainment', 'description': 'A news article about entertainment'},
    {'label': 'Sport', 'description': 'A news article about sport'},
    {'label': 'Business', 'description': 'A news article about business'},
]

"""
In this example, we'll categorize using the label itself, so the first step is to extract the labels as single list and use these as the class description.
"""
class_descriptions = [topics['description'] for topics in topics]

"""
Then embed each topic label using the create_embeddings custom function that makes a call to the OpenAI embedding model.
"""
class_embeddings = create_embeddings(class_descriptions)

"""
The first step is to combine the headline and keyword information into a single string that we can embed.
"""
article = {
    "headline": "How NVIDIA GUPs Could Decide Who Wins the AI Race",
    "keywords": ["ai", "business", "computers"]
}

"""
We do this by defining a custom function that uses and F-string to concatenate the headline and keywords inside a nicely formatted string.
"""
def create_article_text(article):
    return f"""
    Headline: {article['headline']} 
    Keywords: {', '.join(article['keywords'])}"""

article_text = create_article_text(article)
"""
Finally, we can embed the text by calling create_embeddings again, remembering to zero-index the list returned so we have a single list of numbers
"""
article_embedding = create_embeddings(article_text)[0]

"""
Now that we bave the embeddings, it's time for the cosine distances calculation. This is a modified version of the find_n_closest custom function from earlier in the course, where instead of returning n results, we only want one: the nearest label. This means that instead of sorting by distance, we can find the minimum using the min function. Calling this function will return the distance and index of this label.
"""
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])

closest = find_closest(article_embedding, class_embeddings)

"""
Finally, we can use this index to subset the topics dictionary and extract the label. Printing the result, returns the Business label.
"""
label = topics[closest['index']]['label']
print(label)

"""
Limitation:
    Class descriptions lacked sufficient detail
"""
