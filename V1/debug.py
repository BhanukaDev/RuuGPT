from NLPEngineV1 import encodeSentence
import torch

wordIndices = encodeSentence("Hello World!")
print(wordIndices)
