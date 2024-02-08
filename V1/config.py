import json
import numpy as np

def getallTags():
    tags = []

    try:
        with open('V1/dataset.json') as json_file:
            intents = json.load(json_file)
            for intent in intents['intents']:
                for tag in intent['tags']:
                    if tag not in tags:
                        tags.append(tag.lower())
    except Exception as e:
        print("An Error Occured When reading Tags: ", e)

    return tags

tags = getallTags()

def getDataset():
    dataset = []

    try:
        with open("V1/dataset.json") as file:
            intents = json.load(file)
            for intent in intents['intents']:
                tagsPosArray = np.zeros(len(tags),dtype=np.int32)
                for indx,tag in enumerate(tags):
                    if(tag in intent['tags']):
                        tagsPosArray[indx] = 1
                    
                data = (intent['text'], np.array(tagsPosArray))
                dataset.append(data)
    except Exception as e:
        print("An Error Occured When reading Dataset: ", e)

    return dataset
