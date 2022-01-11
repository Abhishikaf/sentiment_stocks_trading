import pandas as pd

# returns a dataframe with rolling average columns. moving average numbers were chosen from reported amounts of the most common ones used by other traders. 5, 10, 20, 50, 100, and 200, plus fibonacci numbers in the same range.
def return_rolling_averages(dataframe):
    windows = [2, 3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200]
    for w in windows:
        dataframe[w]=dataframe["close"].rolling(w).mean()

# # returns a dataframe with signals for each time a lower moving average crosses a higher length rolling average. the result is when a lower one passes a longer one, return signal plus one. when a lower length average dips below a longer one, return signal negative one.
def return_crossovers(dataframe):
    columns = dataframe.columns
    cross_df = pd.DataFrame(index=dataframe.index)
    for col in range(len(dataframe.columns)):
        for col2 in (range(col+1, len(dataframe.columns))):
            cross_df[str(columns[col]) + " to " + str(columns[col2])] = ((dataframe[columns[col2]] < dataframe[columns[col]]) & ((dataframe.shift()[columns[col2]] > dataframe.shift()[columns[col]]))) * 1 - ((dataframe[columns[col2]] > dataframe[columns[col]]) & ((dataframe.shift()[columns[col2]] < dataframe.shift()[columns[col]]))) * 1
    return cross_df