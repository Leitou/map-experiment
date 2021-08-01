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
