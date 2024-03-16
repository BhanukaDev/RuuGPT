import onnxruntime
import torch
import torch.nn as nn
from NLPEngineV1 import encodeSentence
import numpy as np
from config import tags

# Load the ONNX model
onnx_model_path = "V1/onnxs/model6.onnx"
model = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data
sentence = "I want to see elephants near beaches"
wordIndices = torch.tensor(encodeSentence(sentence),dtype=torch.int32).reshape(1,-1)

# Run inference
output = model.run([], {"input": wordIndices.numpy()})
# Print the output
print(output)

# Print the tags
probs = output[0][0]

print("Related Tags: ")
if(np.max(probs) < 0.7):
    print("No tag found!")
for idnx,prob in enumerate(probs):
    if prob > 0:
        print(tags[idnx], ":", prob)
