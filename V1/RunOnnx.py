import onnxruntime
import torch
from NLPEngineV1 import getVocabSize, encodeSentence
from config import tags

# Load the ONNX model
onnx_model_path = "model.onnx"
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Get input and output names
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# Finding duplicate tags
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        if tags[i] == tags[j]:
            print(f"Duplicate at {i}:{tags[i]} and {j}:{tags[j]}")

while True:
    sentence = input("You: ")
    if sentence == "quit" or sentence == "q":
        break

    # Encode the sentence
    wordIndices = torch.tensor(encodeSentence(sentence), dtype=torch.int64).reshape(
        1, -1
    )

    # Run the ONNX model
    outputs = onnx_session.run([output_name], {input_name: wordIndices.numpy()})
    probs = torch.sigmoid(torch.tensor(outputs[0]))

    print("Related Tags: ")
    results = []
    for idnx, prob in enumerate(probs[0]):
        results.append((tags[idnx], prob.item()))

    results.sort(key=lambda x: x[1], reverse=True)
    for tag, prob in results[:5]:
        print(f"{tag}: {prob * 100:.2f}%")
    print("")
