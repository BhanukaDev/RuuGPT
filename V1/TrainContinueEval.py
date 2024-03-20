import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence, getVocabSize
from config import getDataset, tags

import time

startTime = time.time()

vocab_size = getVocabSize()
embedding_dim = 50

hidden_size = 18
output_size = len(tags)
dropout = 0.5

numEpochs = 500


modelPath = input("Enter model path: ")  # Old model path
modelPath = "V1/models/" + modelPath
if not modelPath.endswith(".pth"):
    modelPath += ".pth"

newModelPath = input("Enter new model path: ")  # New model path
newModelPath = "V1/models/" + newModelPath
if not newModelPath.endswith(".pth"):
    newModelPath += ".pth"

numEpochs = int(input("Enter number of epochs: "))  # Number of epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(modelPath)
model = RuuGPTV1(
    checkpoint["vocab_size"],
    checkpoint["embedding_dim"],
    checkpoint["hidden_size"],
    checkpoint["output_size"],
    checkpoint["dropout"],
)
model.load_state_dict(checkpoint["state_dict"])

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

endTime = time.time()

print("Training complete!")
print("Model saved at", newModelPath)
print(f"Time taken: {endTime - startTime:.2f} seconds")
