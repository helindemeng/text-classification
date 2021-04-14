import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(74963, 300)
        self.rnn = nn.LSTM(300, 128, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, c_n) = self.rnn(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        x = self.linear(x.reshape(x.shape[0], -1))
        return x


if __name__ == '__main__':
    data = torch.randint(0, 74963, size=(5, 300))
    net = Model()
    out = net(data)
    print(out)
