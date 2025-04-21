# Meet Robo: your Python buddy

import io
import random
import string
import warnings
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

# Suppress warnings
warnings.filterwarnings("ignore")

# Read the knowledge base
with open("chatbot.txt", "r", encoding="utf8", errors="ignore") as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = sent_tokenize(raw)
word_tokens = word_tokenize(raw)

# Lemmatization setup
lemmer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(word_tokenize(text.lower().translate(remove_punct_dict)))


# Greetings
GREETING_INPUTS = (
    "hello",
    "hi",
    "greetings",
    "sup",
    "what's up",
    "hey",
    "heyy",
    "helloo",
)
GREETING_RESPONSES = [
    "Hi!",
    "Hey!",
    "*nods*",
    "Hello there!",
    "Hi, how can I help?",
    "I'm glad you're talking to me!",
]


def smart_greeting(sentence):
    sentence = sentence.lower()
    for greeting in GREETING_INPUTS:
        if re.search(rf"\b{greeting}\b", sentence):
            return random.choice(GREETING_RESPONSES)


# Generate response using TF-IDF + cosine similarity
def find_best_match(user_input):
    temp_sent_tokens = sent_tokens.copy()
    temp_sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(temp_sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    return temp_sent_tokens[idx] if req_tfidf != 0 else None


def get_response(user_input):
    greeting_resp = smart_greeting(user_input)
    if greeting_resp:
        return greeting_resp

    matched_sentence = find_best_match(user_input)
    if matched_sentence:
        return f"Here's what I found: {matched_sentence}"
    else:
        return "I'm sorry! I didn't quite catch that. Can you rephrase?"


# Chat loop
def chat():
    print("ROBO: Hello! I'm Robo. Ask me anything about Python. Type 'bye' to exit.")
    while True:
        user_input = input("You: ").strip().lower()

        if not user_input:
            print("ROBO: Please say something!")
            continue

        if any(word in user_input for word in ["bye", "goodbye", "exit", "see you"]):
            print("ROBO: Bye! Take care.")
            break

        if user_input in ("thanks", "thank you"):
            print("ROBO: You're welcome!")
            break

        print("ROBO:", get_response(user_input))


# Run it
if __name__ == "__main__":
    chat()
