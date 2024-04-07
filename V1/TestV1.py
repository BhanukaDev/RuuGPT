import torch
import torch.nn as nn
from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import getVocabSize, encodeSentence
from config import tags


modelPath = input("Enter Model Path : ")
if not modelPath.endswith(".pth"):
    modelPath += ".pth"

modeldata = torch.load("V1/models/" + modelPath)

vocab_size = modeldata["vocab_size"]
embedding_dim = modeldata["embedding_dim"]

hidden_size = modeldata["hidden_size"]
output_size = modeldata["output_size"]
dropout = modeldata["dropout"]


model = RuuGPTV1(vocab_size, embedding_dim, hidden_size, output_size, dropout)

model.load_state_dict(modeldata["state_dict"])

model.eval()


# Finding duplicate tags
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        if tags[i] == tags[j]:
            print(f"Duplicate at {i}:{tags[i]} and {j}:{tags[j]}")


with torch.no_grad():

    while True:
        results = []
        sentence = input("You: ")
        if sentence == "quit" or sentence == "q":
            break
        wordIndices = torch.tensor(encodeSentence(sentence), dtype=torch.int32).reshape(
            1, -1
        )
        output = model(wordIndices)
        probs = torch.sigmoid(output)
        print("Related Tags: ")
        # if(probs[0].max() < 0.6):
        #     print("No tag found!")
        for idnx, prob in enumerate(probs[0]):
            if prob > 0.01:
                results.append((tags[idnx], prob.item()))
        results.sort(key=lambda x: x[1], reverse=True)

        results = results[:3]

        # if results[0][1] > sum([x[1] for x in results[1:]]):
        #     temp = results[0]
        #     results.clear()
        #     results.append(temp)
        # elif results[0][1] + results[1][1] > sum([x[1] for x in results[2:]]):
        #     temp = results[:2]
        #     results.clear()
        #     results.extend(temp)
        # elif results[0][1] + results[1][1] + results[2][1] > sum(
        #     [x[1] for x in results[3:]]
        # ):
        #     temp = results[:3]
        #     results.clear()
        #     results.extend(temp)

        for tag, prob in results:
            print(f"{tag}: {prob*100:.2f}%")
            print("")
