import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence,getVocabSize
from config import getDataset,tags

vocab_size = getVocabSize()
embedding_dim = 100

hidden_size = 18
output_size = len(tags)
dropout = 0.5

numEpochs = 1000

modelPath = input("Enter model path: ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# check whether model with such path exists
try:
    modeldata = torch.load(modelPath)
    print(f"Loading model from {modelPath}")
    model = RuuGPTV1(vocab_size,embedding_dim,modeldata['hidden_size'],modeldata['output_size'],modeldata['dropout'])
    model.load_state_dict(modeldata['state_dict'])

    op = input("Make new version of the model? (y/any): ")
    if op == 'y':
        modelPath = input("Enter new model path: ")

    model.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=modeldata['embedding_dim'])    
    model.fc = nn.Linear(in_features=modeldata['hidden_size'], out_features=len(tags))

except Exception as e:
    print(f"Creating new model at {modelPath}. because an error occured:", e)
    model = RuuGPTV1(vocab_size,embedding_dim,hidden_size,output_size,dropout)


model.to(device)

try:
    numEpochs = int(input("Enter number of epochs: "))
except:
    print("Invalid input, using default value of 1000") 


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = getDataset()

for epoch in range(numEpochs):
    for sentence, tagsPosArray in dataset:
        wordIndices = torch.tensor(encodeSentence(sentence),dtype=torch.int32).reshape(1,-1)
        wordIndices = wordIndices.to(device)
        output = model(wordIndices)

        tagsPosArray = np.array(tagsPosArray)
        tagsPosTensor = torch.tensor(tagsPosArray, dtype=torch.float32).reshape(1, -1)
        tagsPosTensor = tagsPosTensor.to(device)
        

        loss = criterion(output, tagsPosTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item()*100:.4f}%')

data ={
    "vocab_size":vocab_size,
    "embedding_dim":embedding_dim,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "dropout":dropout,
    "state_dict":model.state_dict()
    
}

torch.save(data, modelPath)
