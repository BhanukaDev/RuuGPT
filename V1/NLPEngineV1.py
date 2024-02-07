import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import words
import string

stopwords = stopwords.words('english') + list(string.punctuation)
allwords = words.words()
maxLength = 500

def tokenise(text):
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

def stem(tokens):
    ps = PorterStemmer()
    return [ps.stem(word) for word in tokens]

def getVocabId(word):
    return allwords.index(word)+1 if word in allwords else 0

def sentenceToIds(sentence):
    tokens = tokenise(sentence)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    tokens = tokens[:maxLength]
    ids = [getVocabId(token) for token in tokens]
    return ids

def getVocabSize():
    return len(allwords)
