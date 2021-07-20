from torch import nn


def MLP(in_features: int, hidden_size: int = 256, out_classes: int = 1):
    return nn.Sequential(
        nn.Linear(in_features, hidden_size),  # bias=True is default
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, out_classes),
    )


def AE(in_features: int, hidden_size: int = 32):
    return nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, in_features),
        nn.ReLU()
    )


# TODO: decide on preprocessing of data to have input features in ranges [0,1] for efficiency (divide cols by max value)
class AutoEncoder(nn.Module):
    def __init__(self, K):
        super(AutoEncoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(K, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, K),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.encode(x)
        z = self.decode(z)
        return z
