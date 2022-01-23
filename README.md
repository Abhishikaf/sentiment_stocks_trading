# Algorithmic and Neural Trading Tools with ESG-Focused Sentiment Analysis

Highly rated ESG (Environmental, Social, and Governance) stocks are attractive to many investors for moral reasons, but many may be adverse to risk or want to reduce the risk in their portfolio.

This application tests algorithms and trading strategies including a large amount of moving averages, simple moving averages crossover signals, and machine learning sentiment analysis using neural networks. The simple moving average crossover portion all of the  most commonly used moving average periods crossover signals as trading strategies. The neural portion uses over 200 input signals including percent change of moving averages and close prices, crossover signals, weighted crossover signals, the sum of crossover signals, the sum of weighted crossover signals, and a number of sentiment sources pulled from twitter, google news, and google trends using Natural Language Processing algorithms. A shallow and a deep neural network are both tested on each stock that is picked. Results of each test are stored and saved on a page to view and compare.


---
## Technologies and Installation Guide:

This project uses python 3.7 along with the following packages:

Please verify you have installed the following packages:
 
- pandas
- numpy
- tensorflow
- sklearn
- alpaca-trade-api
- streamlit
- plotly
- Ipython
- wordcloud
- matplotlib
- streamlit

Please acquire API keys from Alpaca.

You will need to create a text file in the program folder labeled `.env` and containing your API key and API secret key in the following format:

```ALPACA_API_KEY = "YOUR-ALPACA-API-KEY-HERE"```
```ALPACA_SECRET_KEY = "YOUR-ALPACA-SECRET-KEY-HERE"```

###Additional functionality
Additional functionality is available if you would like to accumulate sentiment data. CSV files for several stocks are provided with a  small recent set. To get more, you will need to install the following packages: 

- flair
- pygooglenews
- newspaper3k
- nltk
- textblob
- segtok

###Rapidapi:

You  will need an API key for RapidAPI.

(https://rapidapi.com/socialminer/api/twitter32/)

In order to gather additional sentiment signals, you will need to have jupyter notebook and be able to run code in the following files:



---

## Usage and description of some methods:

Please run:

```
streamlit run streamlit_app.py
```

Some functionality relies on CSV files which were compiled from multiple API calls. If you would like to use the program to analyze more stocks, you will need to run these API functions and compile CSV files, or pay subscriptions to allow larger API  call requests.

### Calculating Relevance score
We use Similarity  scores to determine how similar each headline is compared to the rest of the headlines for that particular day.
Given two headlines, $A$ and $B$, the Jaccard Similarity Score is calculated as:
$$
\text{Jaccard Similarity Score} = \frac{A\cap B}{A \cup B}
$$
Simply put, the numerator is the number of words that **are common across both headlines** and the denominator is the **total number of words in both headlines**. Keep in mind that the words here refer to **unique words**.
The higher the score, the more relevant the headline is for that particular day. This means that the events mentioned in the higher scoring headlines are likely to be larger events.
Since stock markets are sensitive to current affairs, larger events are likely to affect the stock market more.
Using the relevance score, we add weightage to the sentiment score for each headline.

###VADER
VADER ( Valence Aware Dictionary for Sentiment Reasoning) is used for text sentiment analysis of unlabelled data, it is sensitive to both polarity (positive/negative) and intensity (strength) of emotion shown. It is available under the library in the NLTK package.
VADER sentimental analysis is dependent upon a dictionary that maps lexical features to emotion intensities better known as sentiment scores. The sentiment score is obtained by summing up the intensity of each word in the text.

TextBlob is a Naive Bayes Analyzer based text sentiment analysis library. It is a simple API for natural language processing (NLP) tasks such as POS tagging, name extraction, sentiment analysis, translation, classification, and more. Textblob sentiment analyzer returns two properties for a given input sentence: Polarity and Subjectivity. Subjectivity is also a float that lies in the range of [0,1] and refers to opinion, emotion, or judgment.

###Flair
(https://github.com/flairNLP/flair)

A python library that uses natural language processing for sentiment analysis. It has a pre-trained sentiment analysis module.

---
## Applications:

LEGAL DISCLAIMER: This program is for eductional purposes. It is not financial advice.

The program is intended to test algorithms and trading strategies that inlcude moving averages, simple moving average crossover signals, and machine learning sentiment analysis from various sources with neural networks.

---

## Analysis:

One will find by playing around with these or any algorithmic machine learning algorithm for trading a wide variety of results. These can include doing worse than the market, losing less than the market, gaining less than the market, doing the same as the market, or gaining more than the market. It will be found that sometimes a strategy appears to be doing better or worse than the market for a period and then the situation is reversed. The most interesting thing about these models is finding out what circumstances they beat the market, or at least provide a lower risk gain than the market.

---
## Future Development:

- Add functionality to account for stock splits.
- Add error messages if stock data does not go back to the begin date or API call is refused.
- Add functionality to add dividends paid out while stock is held to money supply in evaluation functions.
- Run SMA Crossover function on many stocks with more crossover combinations and see what are the best short/long window combinations.
- Implement Keras Tuner.
- Implement early stopping of neural fit function to avoid overfit.
- Implement multiple runs of neural fit to try to get highest validation accuracy.
- Add model for 3-neuron output to predict buy, hold, and sell signals.
- Implement running neural fit on a wide amount of stocks to see which ones can currently be fitted with higher validation accuracy.
- Run backtests where the neural fit is done daily including that current day's information.
- Allow user to enter test and train lengths.
- Automatically try multiple test and train lengths of time to see which ones are producing better results.
- Implement adaptive boosting by oversampling the incorrectly guessed signals to train a second model.
- Allow user to enter a whole portfolio of information.
- Implement more API functions for sentiment and search trend signals.
- Add function for paid API usage to get more sentiment data faster.
- Allow user to enter additional terms to get sentiment and trend data.
- Add moving average and percent change signals of sentiments and trends.
- Filter sentiment and trend data to timeframes starting at the close of the trading day and stopping at the close of the next trading day, instead of by day. Average or sum weekends and holidays with following trading day.
- Train our own NLP neural model based on terms and phrases in financial and target sector reporting.



---

## Additional Acknowledgements:

Stocks lexicon was generated by:
Oliveira, Nuno, Paulo Cortez, and Nelson Areal. "Stock market sentiment lexicon acquisition using microblogging data and statistical measures." Decision Support Systems 85 (2016): 62-73.
Loughran and McDonald Financial Lexicon is available at: (https://sraf.nd.edu/loughranmcdonald-master-dictionary/)

---

## Conception and Coding:

Abhishika Fatehpuria (abhishika@gmail.com)

David Jonathan (djonathan@cox.net)

Dave Thomas (sjufan84@gmail.com)

Preston Hodsman (phodsman@yahoo.com)


---

## License

MIT
