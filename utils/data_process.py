import pandas as pd

# returns a dataframe with rolling average columns
def return_rolling_averages(dataframe):
    windows = [5, 10, 30, 100]
    for w in windows:
        dataframe[w]=dataframe["close"].rolling(w).mean()
    return dataframe

# returns a dataframe with signals for each time a 
def return_crossovers(dataframe):
    