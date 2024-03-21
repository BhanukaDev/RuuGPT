import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import words
import string
from config import getallWords

nltk.download('stopwords')
nltk.download('punkt')

stopwords = stopwords.words("english") + list(string.punctuation)
allwords = getallWords()
maxLength = 500
stemmer = PorterStemmer()


def tokenise(text):
    return word_tokenize(text.lower())


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]


def stem(tokens):
    return [stemmer.stem(word) for word in tokens]


def encode(word):
    word = stemmer.stem(word)
    return allwords.index(word) + 1 if word in allwords else 0


def decode(index):
    return allwords[index - 1] if index > 0 else "Unknown"


def encodeSentence(sentence):
    tokens = tokenise(sentence)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tokens = tokens[:maxLength]
    ids = [encode(token) for token in tokens]
    return ids


def getVocabSize():
    return len(allwords) + 1
