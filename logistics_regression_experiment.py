import pandas as pd

from entities.ExperimentContext import ExperimentContext
from models.LogisticsRegressionModel import LogisticsRegressionModel
from utils.constants import *

data = pd.read_csv('data/jane-street-market-prediction/train_data_small.csv')
num_days = len(data[DATE].unique())

train_set_days = int((2 / 3) * num_days)

train_data = data[data[DATE].isin(range(train_set_days))]
test_data = data[~data[DATE].isin(range(train_set_days))]

model = LogisticsRegressionModel()
context = ExperimentContext(model=model,
                            train_data=train_data,
                            test_data=test_data)

context.run_experiment()



# model.train(features=train_features,
#             weights=train_weights,
#             response=train_response)
#
# train_predicted_actions = model.make_prediction(features=train_features)
#
# train_result = pd.DataFrame()
# train_result[DATE] = train_set[DATE]
# train_result[WEIGHT] = train_weights
# train_result[RESPONSE] = train_response
# train_result[OPTIMAL_TRADE_ACTION] = model.get_optimal_actions(weights=train_weights, response=train_response)
# train_result[ALWAYS_TRADE_ACTION] = model.get_always_trade_actions(weights=train_weights)
# train_result[ACTION] = train_predicted_actions
#
# print(train_result)
#
# calculate_utility_for_df(df=train_result, action_column=ACTION, label='TRAIN_PREDICTED')
# calculate_utility_for_df(df=train_result, action_column=OPTIMAL_TRADE_ACTION, label='TRAIN_OPTIMAL')
# calculate_utility_for_df(df=train_result, action_column=ALWAYS_TRADE_ACTION, label='TRAIN_ALWAYS_TRADE')
#
