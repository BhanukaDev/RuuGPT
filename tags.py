import torch
from V1.RuuGPTV1 import RuuGPTV1
from V1.NLPEngineV1 import encodeSentence
from V1.config import *


# Initialize model 
def init_model():
    modelPath = 'V1/models/model16.pth'  
    modeldata = torch.load(modelPath)

    model = RuuGPTV1(modeldata['vocab_size'], modeldata['embedding_dim'],
                     modeldata['hidden_size'], modeldata['output_size'], modeldata['dropout'])
    model.load_state_dict(modeldata['state_dict'])
    model.eval()
    return model

model = init_model()

def generate_tags(sentence, threshold=0.6):
    results = []
    # Convert the sentence into tensor of word indices
    wordIndices = torch.tensor(encodeSentence(sentence), dtype=torch.int32).reshape(1, -1)
    
    with torch.no_grad():
        # Generate output from the model
        output = model(wordIndices)
        # Apply sigmoid to output to get probabilities
        probs = torch.sigmoid(output)
        # Iterate over the probabilities and tags
        for idx, prob in enumerate(probs[0]):
            # Only add tags with probabilities above the threshold
            if prob.item() > threshold:
                results.append((tags[idx], prob.item()))
    
    # Sort the results by probability in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    return results

