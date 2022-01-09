import alpaca_trade_api as tradeapi
import datetime as dt
from dotenv import load_dotenv
import sys
import os
import pandas as pd

#import environment variables
load_dotenv()
alpaca_api_key = os.getenv('ALPACA_API_KEY')
alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
type(alpaca_api_key)

#Create the Alpaca API object -- Is this necessary?
alpaca = tradeapi.REST(
    alpaca_api_key,
    alpaca_secret_key,
    api_version='v2')

# Alpaca functions
def get_historical_dataframe (ticker, start_date, end_date, timeframe):
    ticker_df = alpaca.get_barset(ticker, timeframe, end = end_date, start = start_date, limit = 1000).df
    ticker_df.reindex(columns = ticker_df.columns)
    return ticker_df
def filter_close_prices(dataframe):
    df_close = pd.DataFrame()
    df_close['close'] = dataframe['close']
    return df_close
def calc_daily_returns(df_close_prices):
    daily_returns = df_close_prices.pct_change().dropna()
    return daily_returns