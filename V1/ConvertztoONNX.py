import torch
import torch.onnx

from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence, getVocabSize

modelPath = input("Enter Model Path : ")

# Load the trained model
trained_model_data = torch.load("V1/models/" + modelPath + ".pth")
vocab_size = trained_model_data["vocab_size"]
embedding_dim = trained_model_data["embedding_dim"]
hidden_size = trained_model_data["hidden_size"]
output_size = trained_model_data["output_size"]
dropout = trained_model_data["dropout"]

# Create an instance of your model
model = RuuGPTV1(vocab_size, embedding_dim, hidden_size, output_size, dropout)
model.load_state_dict(trained_model_data["state_dict"])

# Set the model to evaluation mode
model.eval()

sentence = "I want to visit the beach with mountains in the background."
dummy_input = torch.tensor(encodeSentence(sentence), dtype=torch.int64).reshape(1, -1)
# Export the model to ONNX format
output_path = "model.onnx"
input_names = ["input_ids"]
output_names = ["output"]
dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "output": {0: "batch_size"},
}

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=11,
)

print("Model exported to ONNX format at", output_path)
