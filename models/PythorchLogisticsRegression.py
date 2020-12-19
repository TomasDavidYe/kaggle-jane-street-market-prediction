import torch
from torch.nn import Sigmoid


class PytorchLogisticsRegression:
    def __init__(self, input_size=1, n_epochs=100, learning_rate=1e-3):
        self.__learning_rate = learning_rate
        self.__n_epochs = n_epochs
        self.__weights = torch.tensor([0.0 for _ in range(input_size)], requires_grad=True)
        self.__bias = torch.tensor([0.0], requires_grad=True)
        self.__sigmoid = Sigmoid()

    def get_weights(self):
        return self.__weights

    def get_bias(self):
        return self.__bias

    def forward(self, x):
        return self.__sigmoid(self.__weights.matmul(x) + self.__bias)

    def predict(self, x):
        prob = self.forward(x)
        return (prob >= 0.5).float()

    def fit(self, x, y_true):
        optimizer = torch.optim.SGD([self.__weights, self.__bias], self.__learning_rate)
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
