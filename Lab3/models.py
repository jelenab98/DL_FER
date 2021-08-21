<<<<<<< HEAD
import torch.nn as nn
import torch


class Baseline(nn.Module):
    def __init__(self, embeddings):
        super(Baseline, self).__init__()
        self.embeddings = embeddings
        self.fc1 = nn.Linear(in_features=300, out_features=150)
        self.fc2 = nn.Linear(in_features=150, out_features=150)
        self.fc3 = nn.Linear(in_features=150, out_features=1)

    def forward(self, x):
        e = self.embeddings(x)
        if len(x.shape) == 2:
            dim = 1
        else:
            dim = 0
        e = torch.mean(e, dim=dim)
        e = torch.relu(self.fc1(e))
        e = torch.relu(self.fc2(e))
        e = self.fc3(e)

        return e


class RNNModel(nn.Module):
    def __init__(self, embeddings, encoder, hidden_dim=300):
        super(RNNModel, self).__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.fc1 = nn.Linear(in_features=hidden_dim*2 if encoder.bidirectional else hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(0, 1)
        x, _ = self.encoder(x, None)
        x = torch.relu(self.fc1(x[-1]))
        x = self.fc2(x)
        return x
=======
import torch.nn as nn
import torch


class Baseline(nn.Module):
    def __init__(self, embeddings):
        super(Baseline, self).__init__()
        self.embeddings = embeddings
        self.fc1 = nn.Linear(in_features=300, out_features=150)
        self.fc2 = nn.Linear(in_features=150, out_features=150)
        self.fc3 = nn.Linear(in_features=150, out_features=1)

    def forward(self, x):
        e = self.embeddings(x)
        if len(x.shape) == 2:
            dim = 1
        else:
            dim = 0
        e = torch.mean(e, dim=dim)
        e = torch.relu(self.fc1(e))
        e = torch.relu(self.fc2(e))
        e = self.fc3(e)

        return e


class RNNModel(nn.Module):
    def __init__(self, embeddings, encoder, hidden_dim=300):
        super(RNNModel, self).__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.fc1 = nn.Linear(in_features=hidden_dim*2 if encoder.bidirectional else hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(0, 1)
        x, _ = self.encoder(x, None)
        x = torch.relu(self.fc1(x[-1]))
        x = self.fc2(x)
        return x
>>>>>>> ca8923228a32a1117eff983cbec160e90b72ca02
