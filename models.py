from torch import nn


class MLP(nn.Module):
    def __init__(self, K):
        super(MLP, self).__init__()
        self.linstack = nn.Sequential(
            nn.Linear(K, 512), # bias=True is default
            nn.Sigmoid(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, 1),
            # nn.Sigmoid(),
            # nn.BatchNorm1d(128),
            # nn.Dropout(0.6),
            # nn.Linear(128,1)
        )

    def forward(self, x):
        output = self.linstack(x)
        output = output.view(-1)
        return output