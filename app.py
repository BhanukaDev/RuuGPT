from flask import Flask, request, jsonify
from NLPEngineV1 import encodeSentence
from firestore_db import db
from tags import generate_tags
import torch
from V1.RuuGPTV1 import RuuGPTV1
from V1.config import tags


app = Flask(__name__)


# Load the latest model
modeldata = torch.load("V1/models/latestmodel.pth")

vocab_size = modeldata["vocab_size"]
embedding_dim = modeldata["embedding_dim"]

hidden_size = modeldata["hidden_size"]
output_size = modeldata["output_size"]
dropout = modeldata["dropout"]

model = RuuGPTV1(vocab_size, embedding_dim, hidden_size, output_size, dropout)

model.load_state_dict(modeldata["state_dict"])
model.eval()


@app.route("/generate_tags", methods=["POST"])
def handle_generate_tags():
    data = request.json
    sentence = data.get("sentence")

    # Ensure a sentence was provided
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # Bhanuka's code (Duvin, Here I will return list of tags, use it for query!)
    result_tags = []

    with torch.no_grad():
        wordIndices = wordIndices = torch.tensor(
            encodeSentence(sentence), dtype=torch.int32
        ).reshape(1, -1)

        output = model(wordIndices)
        probs = torch.sigmoid(output)

        for index, prob in enumerate(probs[0]):
            result_tags.append(tags[index], prob.item())
        result_tags.sort(key=lambda x: x[1], reverse=True)

    # Bhanuka's code end

    # Generate tags with probabilities above 60%
    tag_results = generate_tags(sentence, threshold=0.6)

    # Extract just the tag names for querying Firestore
    query_tags = [tag for tag, _ in tag_results]

    # Query Firestore for locations with matching tags
    locations = []
    if query_tags:
        destinations_ref = db.collection("destinations")
        for tag in query_tags:
            docs = destinations_ref.where("tags", "array_contains", tag).stream()
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data["id"] = doc.id  # Include document ID if needed
                # Avoid duplicate locations if they match multiple tags
                if doc_data not in locations:
                    locations.append(doc_data)

    return jsonify(locations)


if __name__ == "__main__":
    app.run(debug=True)
