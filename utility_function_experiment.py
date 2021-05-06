import pandas as pd
from utils.jane_street_math import calculate_utility
from utils.constants import *

train_data_small = pd.read_csv('data/jane-street-market-prediction/train_data_small.csv')
print(train_data_small.head())

train_data_small[ALWAYS_TRADE_ACTION] = train_data_small.index.map(lambda x: 1)
train_data_small[NEVER_TRADE_ACTION] = train_data_small.index.map(lambda x: 0)
train_data_small[OPTIMAL_TRADE_ACTION] = train_data_small.index.map(lambda x: int(train_data_small['resp'][x] > 0))

model_data = train_data_small[
    ['ts_id', DATE, WEIGHT, RESPONSE, OPTIMAL_TRADE_ACTION, ALWAYS_TRADE_ACTION, NEVER_TRADE_ACTION]]
print(DATE)


def calculate_utility_shortcut(action_column):
    return calculate_utility(date=model_data[DATE],
                             weight=model_data[WEIGHT],
                             response=model_data[RESPONSE],
                             action=model_data[action_column],
                             label=action_column)


calculate_utility_shortcut(NEVER_TRADE_ACTION)
calculate_utility_shortcut(ALWAYS_TRADE_ACTION)
calculate_utility_shortcut(OPTIMAL_TRADE_ACTION)
