import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import sentenceToIds,getVocabSize
from config import getDataset,tags

vocab_size = getVocabSize()
embedding_dim = 50

hidden_size = 18
output_size = len(tags)
dropout = 0.5

numEpochs = 1000

modelPath = input("Enter model path: ")

model = RuuGPTV1(vocab_size,embedding_dim,hidden_size,output_size,dropout)

# check whether model with such path exists
try:
    modeldata = torch.load(modelPath)
    print(f"Loading model from {modelPath}")
    model.load_state_dict(modeldata['state_dict'])
    op = input("Make new version of the model? (y/any): ")
    if op == 'y':
        modelPath = input("Enter new model path: ")
    
    
    if (modeldata["output_size"] != output_size):
        model.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

except:
    print(f"Creating new model at {modelPath}")

try:
    numEpochs = int(input("Enter number of epochs: "))
except:
    print("Invalid input, using default value of 1000") 


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = getDataset()

for epoch in range(numEpochs):
    for sentence, tagsPosArray in dataset:
        wordIndices = torch.tensor(sentenceToIds(sentence),dtype=torch.int32).reshape(1,-1)
        output = model(wordIndices)

        tagsPosArray = np.array(tagsPosArray)
        tagsPosTensor = torch.tensor(tagsPosArray, dtype=torch.float32).reshape(1, -1)
        

        loss = criterion(output, tagsPosTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}')

data ={
    "vocab_size":vocab_size,
    "embedding_dim":embedding_dim,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "dropout":dropout,
    "state_dict":model.state_dict()
    
}

torch.save(data, modelPath)
