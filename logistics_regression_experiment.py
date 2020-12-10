import pandas as pd

from entities.ExperimentContext import ExperimentContext
from models.LogisticsRegressionModel import LogisticsRegressionModel
from utils.constants import *

data = pd.read_csv('data/jane-street-market-prediction/train.csv', nrows=10000)
num_days = len(data[DATE].unique())

train_set_days = int((2 / 3) * num_days)

train_data = data[data[DATE].isin(range(train_set_days))]
test_data = data[~data[DATE].isin(range(train_set_days))]

model = LogisticsRegressionModel()
context = ExperimentContext(model=model,
                            train_data=train_data,
                            test_data=test_data)

context.run_experiment()
