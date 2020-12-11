from sklearn.preprocessing import StandardScaler

from models.AbstractModel import AbstractModel
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticsRegressionModel(AbstractModel):

    def __init__(self, columns=None):
        super().__init__()
        self.__model = LogisticRegression()
        self.__scaler = StandardScaler()
        self.__columns = columns


    def fit(self, features, actions):
        self.__model.fit(X=features, y=actions)

    def predict(self, features):
        return self.__model.predict(X=features)

    def transform(self, features):
        result = self.sanitize_features(features)

        if self.__columns is not None:
            result = result[self.__columns]

        if not self._is_trained:
            self.__scaler.fit(X=result)

        result = self.__scaler.transform(result)
        return result
