
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
#import holoviews as hv
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bokeh.palettes import Oranges256 as oranges
from bokeh.sampledata.us_states import data as us_states
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO

# import functions
from utils.AlpacaFunctions import get_historical_dataframe
from utils.data_process import return_rolling_averages
from utils.data_process import return_crossovers
from utils.data_process import return_weighted_crossovers
from utils.data_process import get_ticker_keywords

from utils.neural_functions import shallow_neural
from utils.neural_functions import deep_neural
from utils.eval_functions import sma_crossover_eval
from utils.eval_functions import results_trade_amount_nostop
from utils.eval_functions import results_trade_amount_stops
from utils.eval_functions import buy_or_sell_all_if_available
from utils.eval_functions import buy_or_sell_trade_percent
from utils.sentiment_functions import concat_sentiment

click_count=0

def button_test(layers, epochs, tt_ratio):
    global click_count

    click_count += 1
    #st.write("Execute button clicked", click_count)
    st.write(f"[NOT USED] Neural layers: {layers}   Epochs: {epochs}   Train/test ratio: {tt_ratio}")

def process_ticker():
    global ticker

    st.write("process_ticker:", ticker)

stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)

    wordcloud=wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(4, 4))
    plt.axis('off')

    col1, col2, col3 = st.columns([1,5,1])
    with col2:
        plt.imshow(wordcloud)
        st.pyplot(fig)


##########
## MAIN ##
##########

st.set_page_config(
    layout="wide",
)
# st.title("ALGORITHMIC, MACHINE LEARNING, AND NEURAL TRADING TOOLS WITH ESG SENTIMENT FOCUS")
# Had to use literal path here. Objected to Path('images/Title.jpg')
st.image('images/Title.jpg', use_column_width='auto')

st.sidebar.title("Select a page")
page = st.sidebar.radio('Select a view', options=['Ticker Selection','Algorithm Parameters', 'Model Stats/Summary'], key='1')
st.sidebar.markdown("""---""")

st.sidebar.header("Model Configuration")

if page == 'Ticker Selection':
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""

    ticker = st.selectbox('Select a stock ticker',['GOOG','INTC','MSFT','CRM','BAC','PYPL','AAPL','NVDA','TSLA','VZ'])
    if ticker:
        st.session_state.ticker = ticker
    keywords = get_ticker_keywords(ticker)
    show_wordcloud(keywords)

    today = pd.Timestamp.now(tz="America/New_York")
    start_date = pd.Timestamp(today - pd.Timedelta(days=500)).isoformat()
    end_date = today
    timeframe = '1D'

    # do we need two columns here? 
    # Display SMAs on left, close/sentiment on right?
    left_col, right_col = st.columns(2)
    ticker = st.session_state.ticker
    df = pd.DataFrame(get_historical_dataframe(ticker, start_date, end_date, timeframe)[ticker])
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['SMA100'] = df['close'].rolling(window=100).mean()
    df_close_sma = df[['close', 'SMA20', 'SMA50', 'SMA100']]
    with left_col:
        fig = px.line(df_close_sma,  title=ticker +  " -- " + "Close with SMAs")
        st.plotly_chart(fig)

    df_close_sentiment = df[['close']]
    # Call concat sentiment function
    df_close_sentiment = concat_sentiment(ticker, df_close_sentiment, True, False)
    sentiment_scaler = StandardScaler()
    sentiment_scaler.fit(df_close_sentiment)
    df_close_sentiment_scaled = sentiment_scaler.transform(df_close_sentiment)
    with right_col:
        fig = px.line(df_close_sentiment_scaled,  title=ticker + " -- " + "Close with Twitter and GoogleNews sentiment")
        st.plotly_chart(fig)



if page == 'Algorithm Parameters':
    #st.header("Algorithm Parameters")
    #st.header("Recent ESG Related Search Trends and Sentiment History:")

    if 'ticker' not in st.session_state:
        st.session_state.ticker = "GOOG"

    st.header(f"{st.session_state.ticker}:  Shallow vs. Deep Neural Network")

    # Enable storing dictionary of plots where entries might be:
    #   { model : [plot1, plot2, ...] )
    if 'fig_dict' not in st.session_state:
        st.session_state.fig_dict = {}

    if 'instance' not in st.session_state:
        instance = 1
        st.write("Initializing instance: ", instance)
    else:
        instance = st.session_state.instance 
        st.session_state = instance + 1
        st.write("Updated instance: ", instance)

    #n_layers = st.sidebar.number_input( "Number of Neural Layers", 3, 10, 5, step=1)
    #n_layers = st.sidebar.slider("Neural layers", 3, 10, 5, key="layers")
    #terms st.sidebar.selectbox("Choose Search Terms :", ['climate','green','environmental'])
    st.sidebar.subheader("Select Epochs")
    n_epochs = st.sidebar.slider("Epochs", 20, 300, 20, 20, key="epochs")
    st.sidebar.markdown("""---""")
    
    st.sidebar.subheader("Sentiment sources")
    twitter = st.sidebar.checkbox("Twitter")
    googleNews = st.sidebar.checkbox("GoogleNews")
    if twitter:
        sentiment_sources = "With Twitter sentiment data"
    if googleNews:
        sentiment_sources = "With Google News sentiment data"
    if twitter and googleNews:
        sentiment_sources = "With Twitter and Google News sentiment data"
    st.sidebar.markdown("""---""")
    if twitter or googleNews:
        st.subheader(sentiment_sources)

    # tt_ratio = st.sidebar.select_slider("Train/test Ratio", options=["2/1","3/1","4/1","5/1"], value=("3/1"),key="train_test")

    if st.sidebar.button("Execute"):
        st.write("Instance: ", instance)
        #button_test(n_layers, n_epochs, tt_ratio)

        # Call model functions
        today = pd.Timestamp.now(tz="America/New_York")
        start_date = pd.Timestamp(today - pd.Timedelta(days=500)).isoformat()
        end_date = today
        timeframe = '1D'

        ticker = st.session_state.ticker
        df = pd.DataFrame(get_historical_dataframe(ticker, start_date, end_date, timeframe)[ticker])
        volume_df = pd.DataFrame(df["volume"])
        close_df = pd.DataFrame(df["close"])
        return_rolling_averages(close_df)
        cross_df = return_crossovers(close_df)
        cross_signals = cross_df.sum(axis=1)
        pct_change_df = close_df.pct_change()
        cross_weighted_df = return_weighted_crossovers(close_df, pct_change_df)
        cross_signals_weighted = pd.DataFrame(cross_weighted_df.sum(axis=1))
        signals_input_df = pd.concat([pct_change_df, cross_df, volume_df, pct_change_df, cross_signals, cross_signals_weighted, cross_weighted_df], axis=1)

        X = signals_input_df.dropna()
        y_signal = ((close_df["close"] > close_df["close"].shift()).shift(-1))*1
        y = pd.DataFrame(y_signal).loc[X.index]

        X_train=X[:-100]
        X_test=X[-100:]
        y_train=y[:-100]
        y_test=y[-100:]

        scaler = StandardScaler()
        X_scaler = scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        predictions_shallow = shallow_neural(X_train_scaled, y_train, X_test_scaled, y_test, debug=0)
        predictions_deep = deep_neural(X_train_scaled, y_train, X_test_scaled, y_test, debug=0)

        # Columns can be named e.g. left_header = left_col.text_input("Shallow Neural")
        # then figure title could be e.g. "Actual vs. Test"
        left_col, right_col = st.columns(2)
        with left_col:
            left_fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig_shallow = px.line(predictions_shallow, color_discrete_sequence=['orange'])
            fig_test = px.line(y_test, color_discrete_sequence=['green'])
            left_fig.add_traces(fig_shallow.data + fig_test.data)
            left_fig.update_layout(title_text=ticker + ' Shallow Neural')
            st.plotly_chart(left_fig)

        with right_col:
            right_fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig_deep = px.line(predictions_deep, color_discrete_sequence=['orange'])
            fig_test = px.line(y_test, color_discrete_sequence=['green'])
            right_fig.add_traces(fig_deep.data + fig_test.data)
            right_fig.update_layout(title_text=ticker + ' Deep Neural')
            st.plotly_chart(right_fig)

        st.session_state.fig_dict[instance] = [left_fig, right_fig]

# PLACE HOLDERS (they each return the same dataframes)
#    Which to run, what to plot?

		#get shorter close_df_test dataframe with only the test indexes
#		close_df_test = pd.DataFrame(close_df["close"]).loc[X_test.index]
#		
#		# function to call basic trade test no stops
#		money_on_hand_df, shares_df, value_on_hand_df = results_trade_amount_nostop(start_money, close_df_test, trained_predictions, trade_amount)
#		
#		# function to call basic trade test with stops
#		money_on_hand_df, shares_df, value_on_hand_df = results_trade_amount_stops(start_money, close_df_test, trained_predictions, trade_amount)
#		
#		# alternate strategy. buy all on buy signal, sell all on sell signal, if available.
#		money_on_hand_df, shares_df, value_on_hand_df = buy_or_sell_all_if_available(start_money, close_df_test, trained_predictions)
#		
#		# alternate strategy. spend trade_percent of available money on buy signal, sell trade_percent of available shares on sell signal.
#		money_on_hand_df, shares_df, value_on_hand_df = buy_or_sell_trade_percent(trade_percent, start_money, close_df_test, trained_predictions)

#    terms = st.sidebar.selectbox("Choose Search Terms :", ['climate','green','environmental'])
#    st.write("You selected:", terms);


if page == 'Test Model Performance':
    st.header("Test Model Performance")
    st.image('Screen_Shot_2.png', width=1000)

if page == 'Model Stats/Summary':
    st.header("Model Stats/Summary")
    left_col, middle_col, right_col = st.columns(3)

    tick_size = 12
    axis_title_size = 16

    if 'fig_dict' in st.session_state:
        figures = st.session_state.fig_dict
        for key in figures:
            left_plot = figures[key][0]
            right_plot = figures[key][1]
            st.plotly_chart(left_plot)
            st.plotly_chart(right_plot)

    left_col.subheader("Left Upper Image")
    # left_col.altair_chart(fig, use_container_width=True)
    #left_col.st.image('MC_fiveyear_sim_plot.png', use_container_width=True)
#    with left_col:
#        st.image('MC_fiveyear_sim_plot.png', use_column_width='auto')
#
#    left_col.subheader("Left Lower Image")
#    with left_col:
#        st.image("MC_fiveyear_dist_plot.png", use_column_width='auto')
#
#    middle_col.subheader("Middle data")
#    middle_col.markdown("Lots of text")
#    middle_col.subheader("Middle lower data")
#    middle_col.markdown("More text")
#
#    right_col.subheader("Right data")
#    right_col.markdown("Lots of text")
#    right_col.subheader("Right lower data")
#    right_col.markdown("More text")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
#st.button("Re-run")


