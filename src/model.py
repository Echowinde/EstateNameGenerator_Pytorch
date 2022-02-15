import torch
import torch.nn as nn
from torch.autograd import Variable


class Config:
    def __init__(self, batch_size=64, embedding_dim=8, hidden_dim=64, lr=0.001, epochs=100, generate_len=10, verbose=5):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.generate_len = generate_len
        self.verbose = verbose


class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):  # (词库大小, 词向量维度, LSTM隐藏层维度)
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=True)  # LSTM
        self.linear = nn.Linear(self.hidden_dim, vocab_size)  # 全连接层

    def forward(self, input_, hidden=None):
        batch_size, seq_len = input_.size()  # (batch_size, seq_len)
        if hidden is None:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_dim))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_dim))
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input_)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.contiguous().view(batch_size * seq_len, -1))  # (batch_size, seq_len, hidden_dim)
        return output, hidden
