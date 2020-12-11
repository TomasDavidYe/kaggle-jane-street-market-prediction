import numpy as np
from abc import ABC, abstractmethod

from entities.JaneStreetDataSet import JaneStreetDataSet


class AbstractModel(ABC):
    def __init__(self):
        self._is_trained = False

    def train(self, features, actions_for_training):
        self.fit(features=self.transform(features),
                 actions=actions_for_training)
        self._is_trained = True

    def make_prediction(self, features):
        return self.predict(self.transform(features))

    @staticmethod
    def sanitize_features(features):
        result = features.fillna(0)
        result = result.replace(np.nan, 0)
        result = result.replace('nan')
        return result

    @abstractmethod
    def fit(self, features, actions):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def transform(self, features):
        pass

