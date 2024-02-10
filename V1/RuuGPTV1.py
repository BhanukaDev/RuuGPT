import torch
import torch.nn as nn

class RuuGPTV1(nn.Module):
    def __init__(self, vocab_size,embedding_dim,hidden_size, output_size,dropout):
        super(RuuGPTV1, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        embedded_x = self.embedding(x)
        lstm_out, _ = self.lstm(embedded_x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        output_probs = self.sigmoid(output)
        return output_probs

    