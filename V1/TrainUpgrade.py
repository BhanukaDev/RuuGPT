import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence, getVocabSize
from config import getDataset, tags

numEpochs = 10

modelPath = input("Enter model path: ")  # Old model path
newModelPath = input("Enter new model path: ")  # New model path

modelPath = "V1/models/" + modelPath
if not modelPath.endswith(".pth"):
    modelPath += ".pth"

newModelPath = "V1/models/" + newModelPath
if not newModelPath.endswith(".pth"):
    newModelPath += ".pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(modelPath)
vocab_size = checkpoint["vocab_size"]
embedding_dim = checkpoint["embedding_dim"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]
dropout = checkpoint["dropout"]

print(getVocabSize(), checkpoint["vocab_size"], output_size, len(tags))

if vocab_size != getVocabSize():
    vocab_size = getVocabSize()

if output_size != len(tags):
    output_size = len(tags)

model = RuuGPTV1(vocab_size, embedding_dim, hidden_size, output_size, dropout)

# Initialize the output layer with the correct shape
model.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

# Load the state dictionary, excluding the output layer
model_dict = model.state_dict()
pretrained_dict = {
    k: v
    for k, v in checkpoint["state_dict"].items()
    if k != "fc.weight" and k != "fc.bias"
}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = getDataset()

for epoch in range(numEpochs):
    for sentence, tagsPosArray in dataset:
        wordIndices = torch.tensor(encodeSentence(sentence), dtype=torch.int32).reshape(
            1, -1
        )
        wordIndices = wordIndices.to(device)
        output = model(wordIndices)
        tagsPosArray = np.array(tagsPosArray)
        tagsPosTensor = torch.tensor(tagsPosArray, dtype=torch.float32).reshape(1, -1)
        tagsPosTensor = tagsPosTensor.to(device)
        loss = criterion(output, tagsPosTensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item()*100:.4f}%")

data = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "dropout": dropout,
    "state_dict": model.state_dict(),
}

torch.save(data, newModelPath)
