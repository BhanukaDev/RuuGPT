import torch
from RuuGPTV1 import RuuGPTV1
from NLPEngineV1 import encodeSentence

if __name__ == "__main__":
    # Load model data
    mdata = torch.load("V1/models/model6.pth")

    # Create and load the model
    model = RuuGPTV1(mdata['vocab_size'], mdata['embedding_dim'], mdata['hidden_size'], mdata['output_size'], mdata['dropout'])
    model.load_state_dict(mdata['state_dict'])
    model.eval()

    # Prepare input data
    sentence = "Hello"
    wordIndices = torch.tensor(encodeSentence(sentence), dtype=torch.int32).reshape(1, -1)

    # Set the sequence length explicitly during export
    torch.onnx.export(model, wordIndices, "V1/onnxs/model6.onnx", input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}})
