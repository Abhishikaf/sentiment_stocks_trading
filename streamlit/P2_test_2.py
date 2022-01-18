
import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from bokeh.palettes import Oranges256 as oranges
from bokeh.sampledata.us_states import data as us_states
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image


st.set_page_config(
    layout="wide",
)
st.title("ALGORITHMIC, MACHINE LEARNING, AND NEURAL TRADING TOOLS WITH ESG SENTIMENT FOCUS")

#st.plotly_chart(fig)

#Plots for data pivoted by states

#st.markdown("This application is a Share Price dashboard for Top 5 Gainers and Losers:")




st.sidebar.title("Model Configuration")
#st.sidebar.markdown("")
page = st.sidebar.selectbox('Page', ['Algorithm Parameters', 'Test Model Performance', 'Model Stats/Summary'], key='1')
#st.write("Page selected:", page)

if page == 'Algorithm Parameters':
    #st.header("Algorithm Parameters")
    st.header("Recent ESG Related Search Trends and Sentiment History:")

    time = st.sidebar.selectbox("Choose a Time Period:", ['one','two','three'])
    st.write("You selected:", time);

    terms = st.sidebar.selectbox("Choose Search Terms :", ['climate','green','environmental'])
    st.write("You selected:", terms);
    st.image('MC_fiveyear_sim_plot.png', use_column_width='auto')


if page == 'Test Model Performance':
    st.header("Test Model Performance")
    st.image('Screen_Shot_2.png', use_column_width='auto')

if page == 'Model Stats/Summary':
    st.header("Model Stats/Summary")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")


