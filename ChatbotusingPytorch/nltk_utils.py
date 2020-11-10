import numpy as np
import nltk
 #nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenised_sent,all_words):
    tokenised_sent=[stem(w) for w in tokenised_sent]
    bag=np.zeros(len(all_words),dtype=np.float)
    for idx,w in enumerate(all_words):
        if w in tokenised_sent:
            bag[idx]=1.0
    return bag





