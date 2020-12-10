import math
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix

from utils.constants import *
from utils.plot_utils import plot_roc_curve


def calculate_utility_bulk(data: pd.DataFrame, label: str):
    calculate_utility_for_df(df=data, action_column=ACTION, label=f'{label}_PREDICTED')
    calculate_utility_for_df(df=data, action_column=OPTIMAL_TRADE_ACTION, label=f'{label}_OPTIMAL')
    calculate_utility_for_df(df=data, action_column=ALWAYS_TRADE_ACTION, label=f'{label}_ALWAYS_TRADE')


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
    print(f'{label}: NUM_TRADING_OPPORTUNITIES = {len(df[ACTION])} ')
    print(f'{label}: NUM_OF_TRADES = {len(df[df[ACTION] == 1])}')
    print(f'{label}: SUM_PROFIT = {sum_p} ')
    print(f'{label}: STD_PROFIT = {std_p} ')
    print(f'{label}: T_COEFFICIENT = {t_coefficient} ')
    print(f'{label}: U_COEFFICIENT = {u_coefficient} ')
    print(f'{label}: UTILITY = {utility} ')

    print(f'--------------------CALCULATING UTILITY END {label}-------------------\n')
    return [sum_p, std_p, t_coefficient, u_coefficient, utility]


def calculate_accuracy(y_true, y_pred, label):
    print(f'------------------Performance Analysis for {label} SET Start--------------------')
    print(f'Accuracy = {accuracy_score(y_true, y_pred)}')
    print(f'F1 Score = {f1_score(y_true, y_pred)}')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    area_under_roc_curve = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, area_under_roc_curve, label)
    print(f'Area under ROC curve = {area_under_roc_curve}')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'TP = {tp}, FP = {fp}')
    print(f'FN = {fn}, TN = {tn}')
    print(f'------------------Performance Analysis for {label} SET End----------------------')
