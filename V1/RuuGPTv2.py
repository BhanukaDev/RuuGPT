import torch
import torch.nn as nn


class RuuGPTV2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        output_size,
        dropout,
        num_layers=1,
        bidirectional=False,
    ):
        super(RuuGPTV2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded_x = self.embedding(x)
        embedded_x = self.dropout(embedded_x)
        output, _ = self.lstm(embedded_x)
        output = self.dropout(output)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
