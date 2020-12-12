from torch import nn


class MLP(nn.Module):

    def __init__(self, layers=10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, layers),
            nn.ReLU(),
            nn.Linear(layers, layers),
            nn.ReLU(),
            nn.Linear(layers, 1)
        )

    def forward(self, x):
        tmp = x.reshape(x.size()[0], 1)
        evaluated = self.layers(tmp)
        reshaped = evaluated.reshape(1, evaluated.size()[0])[0]
        return reshaped
