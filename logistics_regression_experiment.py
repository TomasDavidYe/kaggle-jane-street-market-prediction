import pandas as pd

from entities.ExperimentContext import ExperimentContext
from models.LogisticsRegressionModel import LogisticsRegressionModel
from utils.constants import *

data = pd.read_csv('data/jane-street-market-prediction/train.csv', nrows=100000)
num_days = len(data[DATE].unique())

train_set_days = int((1 / 2) * num_days)

train_data = data[data[DATE].isin(range(train_set_days))]
test_data = data[~data[DATE].isin(range(train_set_days))]

model = LogisticsRegressionModel(columns=[f'feature_{i}' for i in range(50)])
context = ExperimentContext(model=model,
                            train_data=train_data,
                            test_data=test_data)

context.run_experiment()
