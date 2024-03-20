from flask import Flask, request, jsonify
from firestore_db import db
from tags import generate_tags


app = Flask(__name__)

@app.route('/generate_tags', methods=['POST'])
def handle_generate_tags():
    data = request.json
    sentence = data.get('sentence')
    
    # Ensure a sentence was provided
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    # Generate tags with probabilities above 60%
    tag_results = generate_tags(sentence, threshold=0.6)

    # Extract just the tag names for querying Firestore
    query_tags = [tag for tag, _ in tag_results]

    # Query Firestore for locations with matching tags
    locations = []
    if query_tags:
        destinations_ref = db.collection('destinations')
        for tag in query_tags:
            docs = destinations_ref.where('tags', 'array_contains', tag).stream()
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data['id'] = doc.id  # Include document ID if needed
                # Avoid duplicate locations if they match multiple tags
                if doc_data not in locations:
                    locations.append(doc_data)

    return jsonify(locations)


if __name__ == '__main__':
    app.run(debug=True)