import torch
import torch.nn as nn


class CBOWModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_idx):
        embeds = self.embedding(input_idx).mean(1)  # (Batch_size x Context_size x Embed_size).mean(1)
        # print(embeds.shape)  # (Batch_size x Embed_size)
        x = self.linear1(embeds)
        x = self.log_softmax(x)
        return x

    def get_embedding_from_idx(self, idx):
        return self.embedding(idx).detach()

    def get_idx_from_embedding(self, embed):
        dist = torch.norm(self.embedding.weight.data - embed, dim=1)
        return torch.argmin(dist)

