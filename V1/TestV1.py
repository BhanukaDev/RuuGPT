import torch
import torch.nn as nn
from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import getVocabSize, sentenceToIds
from config import tags

modeldata = torch.load('V1/models/model6.pth')

vocab_size = modeldata['vocab_size']
embedding_dim = modeldata['embedding_dim']

hidden_size = modeldata['hidden_size']
output_size = modeldata['output_size']
dropout =  modeldata['dropout']


model = RuuGPTV1(vocab_size,embedding_dim,hidden_size,output_size,dropout)

model.load_state_dict(modeldata['state_dict'])

model.eval()

with torch.no_grad():
    while True:
        sentence = input("You: ")
        if sentence == "quit" or sentence =="q":
            break
        wordIndices = torch.tensor(sentenceToIds(sentence),dtype=torch.int32).reshape(1,-1)
        output = model(wordIndices)
        probs = torch.sigmoid(output)
        print("Related Tags: ")
        if(probs[0].max() < 0.7):
            print("No tag found!")
        for idnx,prob in enumerate(probs[0]):
            if prob > 0.7:
                print(tags[idnx], ":", prob)
        print("")