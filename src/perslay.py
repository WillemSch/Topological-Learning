import torch as nn

class PersLay(nn.Module):
    def __init__(self):
        self.weight = nn.Linear
        self.phi = nn.Linear

    def forward(self, x):
        weight = self.weight.forward(x)
        out = weight * self.phi.forward(x)