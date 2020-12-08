import pandas as pd

from models.LogisticsRegressionModel import LogisticsRegressionModel
from utils.constants import *
from utils.math import calculate_utility, calculate_utility_for_df

data = pd.read_csv('data/jane-street-market-prediction/train_data_small.csv')
num_days = len(data[DATE].unique())

train_set_days = int((2 / 3) * num_days)

train_set = data[data[DATE].isin(range(train_set_days))]
test_set = data[~data[DATE].isin(range(train_set_days))]
feature_columns = list(filter(lambda x: 'feature_' in x, data.columns))

train_weights = train_set[WEIGHT]
train_response = train_set[RESPONSE]
train_features = train_set.filter(regex='^feature_', axis=1)

test_weights = test_set[WEIGHT]
test_response = test_set[RESPONSE]
test_features = test_set.filter(regex='^feature_', axis=1)

model = LogisticsRegressionModel()
model.train(features=train_features,
            weights=train_weights,
            response=train_response)

train_predicted_actions = model.make_prediction(features=train_features)

train_result = pd.DataFrame()
train_result[DATE] = train_set[DATE]
train_result[WEIGHT] = train_weights
train_result[RESPONSE] = train_response
train_result[OPTIMAL_TRADE_ACTION] = model.get_optimal_actions(weights=train_weights, response=train_response)
train_result[ALWAYS_TRADE_ACTION] = model.get_always_trade_actions(weights=train_weights)
train_result[ACTION] = train_predicted_actions

print(train_result)

calculate_utility_for_df(df=train_result, action_column=ACTION, label='TRAIN_PREDICTED')
calculate_utility_for_df(df=train_result, action_column=OPTIMAL_TRADE_ACTION, label='TRAIN_OPTIMAL')
calculate_utility_for_df(df=train_result, action_column=ALWAYS_TRADE_ACTION, label='TRAIN_ALWAYS_TRADE')

