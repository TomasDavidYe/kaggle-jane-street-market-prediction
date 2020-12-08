import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# plotting stuff
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

colorMap = sns.light_palette("blue", as_cmap=True)
import dabl
import datatable as dt

import missingno as msno
import warnings

warnings.filterwarnings('ignore')

# Source -> https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance

train_data_datatable = dt.fread('../input/jane-street-market-prediction/train.csv')
train_data = train_data_datatable.to_pandas()

