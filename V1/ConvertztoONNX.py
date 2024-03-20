import torch

from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence

import torch.onnx

mdata = torch.load("V1/models/model17.pth")

# Create and load the model
model = RuuGPTV1(
    mdata["vocab_size"],
    mdata["embedding_dim"],
    mdata["hidden_size"],
    mdata["output_size"],
    mdata["dropout"],
)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor with a correct shape (batch_size=1)
sentence = "Hello"
dummy_input = torch.tensor(encodeSentence(sentence), dtype=torch.int32).reshape(1, -1)

# Export the model to ONNX format
output_file = "V1/onnxs/model17.onnx"
input_names = ["input_sequence"]
output_names = ["output_probs"]
torch.onnx.export(
    model, dummy_input, output_file, input_names=input_names, output_names=output_names
)
