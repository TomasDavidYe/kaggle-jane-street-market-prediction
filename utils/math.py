import math
import pandas as pd

from utils.constants import *


def calculate_utility_for_df(df, action_column, label='TRAIN'):
    return calculate_utility(date=df[DATE],
                             weight=df[WEIGHT],
                             response=df[RESPONSE],
                             action=df[action_column],
                             label=label)


def calculate_utility(date, weight, response, action, label='TRAIN'):
    print(f'--------------------CALCULATING UTILITY START {label}-----------------')
    df = pd.DataFrame(index=date.index)
    df[DATE] = date
    df[WEIGHT] = weight
    df[RESPONSE] = response
    df[ACTION] = action

    df[ACTION_PROFIT] = df[WEIGHT] * df[RESPONSE] * df[ACTION]
    grouped_profit = df[[DATE, ACTION]] \
        .groupby(DATE) \
        .sum()

    sum_p = df[ACTION_PROFIT].sum()
    std_p = math.sqrt((df[ACTION_PROFIT] ** 2).sum())
    num_days = len(grouped_profit)

    t_coefficient = (sum_p / std_p) * math.sqrt(250 / num_days)
    u_coefficient = min(max(0, t_coefficient), 6)
    utility = u_coefficient * sum_p

    print(f'{label}: NUM_DAYS = {num_days} ')
    print(f'{label}: SUM_PROFIT = {sum_p} ')
    print(f'{label}: STD_PROFIT = {std_p} ')
    print(f'{label}: T_COEFFICIENT = {t_coefficient} ')
    print(f'{label}: U_COEFFICIENT = {u_coefficient} ')
    print(f'{label}: UTILITY = {utility} ')

    print(f'--------------------CALCULATING UTILITY END {label}-------------------\n')
    return [sum_p, std_p, t_coefficient, u_coefficient, utility]
