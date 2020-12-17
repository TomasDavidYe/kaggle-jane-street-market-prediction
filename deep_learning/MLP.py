import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, learning_rate, n_epochs):
        super(MLP, self).__init__()
        self.__learning_rate = learning_rate
        self.__n_epochs = n_epochs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        tmp = x.reshape(x.size()[1], x.size()[0])
        hidden = self.fc1(tmp)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        reshaped = output.reshape(1, output.size()[0])[0]
        return reshaped


    def predict(self, x):
        prob = self.forward(x)
        return (prob >= 0.5).float()

    def fit(self, x, y_true):
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

