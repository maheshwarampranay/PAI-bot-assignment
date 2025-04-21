# Meet Robo: your friend

# Import necessary libraries
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Suppress warnings
warnings.filterwarnings("ignore")

# First-time setup – download required NLTK data
#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download("omw-1.4")

# Read the corpus
with open("chatbot.txt", "r", encoding="utf8", errors="ignore") as fin:
    raw = fin.read().lower()

# Tokenize into sentences and words
sent_tokens = sent_tokenize(raw)
word_tokens = word_tokenize(raw)

# Lemmatization setup
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(word_tokenize(text.lower().translate(remove_punct_dict)))


# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = [
    "hi",
    "hey",
    "*nods*",
    "hi there",
    "hello",
    "I'm glad you're talking to me!",
]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generate response using TF-IDF + cosine similarity
def response(user_response):
    robo_response = ""
    temp_sent_tokens = sent_tokens.copy()
    temp_sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(temp_sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = "I'm sorry! I didn't understand that."
    else:
        robo_response = temp_sent_tokens[idx]
    return robo_response


# Chat loop
flag = True
print(
    "ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!"
)

while flag:
    user_response = input().strip().lower()

    if not user_response:
        print("ROBO: Please say something!")
        continue

    if user_response in ("bye", "bye!", "goodbye"):
        flag = False
        print("ROBO: Bye! Take care..")
    elif user_response in ("thanks", "thank you"):
        flag = False
        print("ROBO: You're welcome!")
    else:
        greet = greeting(user_response)
        if greet is not None:
            print("ROBO: " + greet)
        else:
            print("ROBO: " + response(user_response))
