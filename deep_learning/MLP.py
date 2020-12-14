import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_layers=1, hidden_layers=10, n_epochs=100, learning_rate=1e-3):
        super(MLP, self).__init__()
        self.__learning_rate = learning_rate
        self.__n_epochs = n_epochs
        self.__layers = nn.Sequential(
            nn.Linear(input_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        tmp = x.reshape(x.size()[1], x.size()[0])
        evaluated = self.__layers(tmp)
        reshaped = evaluated.reshape(1, evaluated.size()[0])[0]
        return reshaped



    def predict(self, x):
        prob = self.forward(x)
        return (prob >= 0.5).float()

    def fit(self, x, y_true):
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), self.__learning_rate)
        loss_fn = torch.nn.BCELoss()

        for epoch in range(1, self.__n_epochs + 1):
            optimizer.zero_grad()
            # Forward pass
            y_pred = self.forward(x)
            # Compute Loss
            loss = loss_fn(y_pred, y_true)

            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()

        self.eval()
