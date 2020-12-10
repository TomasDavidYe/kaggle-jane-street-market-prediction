import pandas as pd
import numpy as np

from entities.JaneStreetDataSet import JaneStreetDataSet
from models.AbstractModel import AbstractModel
from utils.constants import *
from utils.math import calculate_utility_bulk, calculate_accuracy


class ExperimentContext:
    def __init__(self, model: AbstractModel, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.model = model
        self.train_set = JaneStreetDataSet(df=train_data, label='TRAIN_SET')
        self.test_set = JaneStreetDataSet(df=test_data, label='TEST_SET')

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

    def run_experiment(self):
        self.model.train(features=self.sanitize_features(self.train_set.features),
                         actions_for_training=self.get_optimal_actions(weights=self.train_set.weights,
                                                                       response=self.train_set.response))

        self.interpret_results(self.train_set)
        self.interpret_results(self.test_set)

    def interpret_results(self, data_set: JaneStreetDataSet):
        result_df = self.get_result_df(data_set)
        calculate_utility_bulk(data=result_df, label=data_set.label)
        calculate_accuracy(y_true=result_df[OPTIMAL_TRADE_ACTION],
                           y_pred=result_df[ACTION],
                           label=data_set.label)

    def get_result_df(self, data_set: JaneStreetDataSet) -> pd.DataFrame:
        predicted_actions = self.model.make_prediction(features=self.sanitize_features(data_set.features))
        res = pd.DataFrame()

        res[DATE] = data_set.date
        res[WEIGHT] = data_set.weights
        res[RESPONSE] = data_set.response
        res[ACTION] = predicted_actions
        res[ALWAYS_TRADE_ACTION] = self.get_always_trade_actions(weights=data_set.weights)
        res[OPTIMAL_TRADE_ACTION] = self.get_optimal_actions(weights=data_set.weights,
                                                             response=data_set.response)

        return res
