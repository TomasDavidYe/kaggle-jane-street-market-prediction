import numpy as np
import torch


from deep_learning.MLP import MLP
from models.AbstractModel import AbstractModel



class SimpleMLPModel(AbstractModel):
    def __init__(self, num_of_epochs=100):
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


    def predict(self, features):
        return self.__mlp(features)

    def transform(self, features):
        result = self.sanitize_features(features)
        result = torch.from_numpy(result.to_numpy()).float()
        return result
