from models.AbstractModel import AbstractModel
from sklearn.linear_model import LogisticRegression


class LogisticsRegressionModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.__model = LogisticRegression()

    def fit(self, features, actions):
        self.__model.fit(X=features, y=actions)

    def predict(self, features):
        return self.__model.predict(X=features)
