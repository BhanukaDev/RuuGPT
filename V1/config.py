import json
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import torch

stemer = PorterStemmer()


def getallWords():
    allWords = []
    try:
        with open("V1/allWords.json") as file:
            allWords = json.load(file)
    except Exception as e:
        # print("An Error Occured When reading Words: ", e)
        allWords = writeallWords()
    return allWords


def writeallWords():
    allWords = []
    modelInfo = torch.load("V1/models/model22.pth")
    allWords = modelInfo["allwords"]
    with open("V1/allWords.json", "w") as file:
        json.dump(allWords, file)
    return allWords


def getallTags():
    tags = []

    try:
        with open("V1/dataset.json") as json_file:
            intents = json.load(json_file)
            for intent in intents["intents"]:
                for tag in intent["tags"]:
                    if tag.lower() not in tags:
                        tags.append(tag.lower())

    except Exception as e:
        # print("An Error Occured When reading Tags: ", e)
        pass

    return tags


tags = getallTags()


def getDataset():
    dataset = []

    try:
        with open("V1/dataset.json") as file:
            intents = json.load(file)
            for intent in intents["intents"]:
                tagsPosArray = np.zeros(len(tags), dtype=np.int32)
                for indx, tag in enumerate(tags):
                    if tag in intent["tags"]:
                        tagsPosArray[indx] = 1

                data = (intent["text"], np.array(tagsPosArray))
                dataset.append(data)
    except Exception as e:
        # print("An Error Occured When reading Dataset: ", e)
        pass

    return dataset
