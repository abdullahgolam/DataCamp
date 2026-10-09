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
     "keywords": ["space", "exploration", "india"]},
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
print(create_article_text(articles[-1]))
