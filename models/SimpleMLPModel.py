import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from deep_learning.MLP import MLP
from models.AbstractModel import AbstractModel



class SimpleMLPModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.__mlp = MLP()
        self.__loss_fn = torch.nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(params=self.__mlp.parameters(),
                                            lr=0.001)


    def fit(self, features, actions):
        mean_train_losses = []
        mean_valid_losses = []
        valid_acc_list = []
        labels = torch.from_numpy(actions.to_numpy())
        correct = 0
        total = 0
        epochs = 15

        for epoch in range(epochs):
            self.__mlp.train()

            train_losses = []
            valid_losses = []


            self.__optimizer.zero_grad()

            predicted = self.__mlp(features)
            loss = self.__loss_fn(predicted, labels)
            loss.backward()
            self.__optimizer.step()

            train_losses.append(loss.item())


            self.__mlp.eval()


            correct += (predicted == labels).sum().item()
            total += 1

            mean_train_losses.append(np.mean(train_losses))

            accuracy = 100 * correct / total
            valid_acc_list.append(accuracy)
            print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%' \
                  .format(epoch + 1, np.mean(train_losses), np.mean(valid_losses), accuracy))

    def predict(self, features):
        return self.__mlp(features)

    def transform(self, features):
        result = self.sanitize_features(features)
        result = torch.from_numpy(result.to_numpy()).float()
        return result
