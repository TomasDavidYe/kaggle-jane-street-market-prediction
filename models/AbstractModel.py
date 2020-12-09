import numpy as np
from abc import ABC, abstractmethod

from entities.JaneStreetDataSet import JaneStreetDataSet


class AbstractModel(ABC):
    def __init__(self):
        pass

    def train(self, features, actions_for_training):
        self.fit(features=features,
                 actions=actions_for_training)

    def make_prediction(self, features):
        return self.predict(features)

    @abstractmethod
    def fit(self, features, actions):
        pass

    @abstractmethod
    def predict(self, features):
        pass

