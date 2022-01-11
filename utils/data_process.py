import pandas as pd

def return_rolling_averages(dataframe):
    windows = [5, 10, 30, 100]
    for w in windows:
        dataframe[w]=dataframe["close"].rolling(w).mean()
    return dataframe


# def return_crossovers(dataframe)