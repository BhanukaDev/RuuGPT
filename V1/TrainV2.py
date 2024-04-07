import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RuuGPTv2 import RuuGPTV2
from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence, getVocabSize, allwords
from config import getDataset, tags

import time

startTime = time.time()


vocab_size = getVocabSize()
embedding_dim = 50

hidden_size = 18
output_size = len(tags)
dropout = 0.5

numEpochs = 50

modelPath = input("Enter model path: ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = RuuGPTV1(vocab_size, embedding_dim, hidden_size, output_size, dropout)
model = RuuGPTV2(vocab_size, embedding_dim, hidden_size, output_size, dropout)

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
    "tags": tags,
    "allwords": allwords,
}

torch.save(data, modelPath)

endTime = time.time()

print("Training complete!")
print("Model saved at", modelPath)
print(f"Time taken: {endTime - startTime:.2f} seconds")
