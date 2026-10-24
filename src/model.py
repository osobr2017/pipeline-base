import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)