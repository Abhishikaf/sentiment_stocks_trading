import pandas as pd
import numpy as np

def results_trade_amount_nostop(close_df_test, trained_predictions, trade_amount):
    start_money = 10000
    shares = 0
    shares_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=trained_predictions.shape[0]), index=trained_predictions.index)
    money_on_hand = start_money
    
    for day in range(len(trained_predictions)):
        shares_df.iloc[day][0] = shares
        money_on_hand_df.iloc[day][0] = money_on_hand
        value_on_hand_df.iloc[day][0] = (shares * close_df_test.iloc[day]["close"]) + money_on_hand
        if trained_predictions.iloc[day][0] == 0:
            shares -= trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand += trade_amount
        elif trained_predictions.iloc[day][0] == 1:
            shares += trade_amount / close_df_test.iloc[day]["close"]
            money_on_hand -= trade_amount
            
    return money_on_hand_df, shares_df, value_on_hand_df


def sma_crossover_eval(start_money, cross_df, close_df):
    start_money_reset = start_money
    shares_reset = 0

    shares_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)
    money_on_hand_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)
    value_on_hand_df = pd.DataFrame(np.zeros(shape=cross_df.shape), index=cross_df.index, columns = cross_cols)



    for col in cross_cols:
        money_on_hand = start_money_reset
        shares = shares_reset
        for day in range(len(cross_df)):
            shares_df[col].iloc[day] = shares
            money_on_hand_df[col].iloc[day] = money_on_hand
            value_on_hand_df[col].iloc[day] = (shares * close_df.iloc[day]["close"]) + money_on_hand
            if (cross_df[col].iloc[day] == -1) & (shares > 0):
                money_on_hand = shares * close_df.iloc[day]["close"]
                shares = 0
            elif (cross_df[col].iloc[day] == 1) & (money_on_hand > 0):
                shares = money_on_hand / close_df.iloc[day]["close"]
                money_on_hand = 0