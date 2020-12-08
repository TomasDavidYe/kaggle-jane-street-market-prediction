import numpy as np
from abc import ABC, abstractmethod



class AbstractModel(ABC):
    def __init__(self):
        self._train_features = None
        self._train_actions = None

    @staticmethod
    def get_optimal_actions(weights, response):
        return (weights * response > 0).astype('int')

    @staticmethod
    def get_always_trade_actions(weights):
        return (1 - weights * 0).astype('int')

    @staticmethod
    def sanitize_features(features):
        result = features.fillna(0)
        result = result.replace(np.nan, 0)
        result = result.replace('nan')
        return result

    def train(self, features, weights, response):
        self._train_features = self.sanitize_features(features)
        self._train_actions = self.get_optimal_actions(weights=weights, response=response)
        self.fit(features=self._train_features,
                 actions=self._train_actions)

    def make_prediction(self, features):
        return self.predict(self.sanitize_features(features))

    @abstractmethod
    def fit(self, features, actions):
        pass

    @abstractmethod
    def predict(self, features):
        pass

