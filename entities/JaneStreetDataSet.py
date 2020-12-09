import pandas as pd

from utils.constants import *


class JaneStreetDataSet:
    def __init__(self, df: pd.DataFrame, label: str):
        self.feature_columns = list(filter(lambda x: 'feature_' in x, df.columns))
        self.ts_id = df[TS_ID]
        self.date = df[DATE]
        self.weights = df[WEIGHT]
        self.response = df[RESPONSE]
        self.features = df.filter(regex='^feature_', axis=1)
        self.label = label


