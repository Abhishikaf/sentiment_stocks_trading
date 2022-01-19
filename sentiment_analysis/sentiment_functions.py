import pandas as pd
from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')
# Import flair Sentence to process input text
from flair.data import Sentence
import regex as re
from pygooglenews import GoogleNews
from newspaper import Article
from newspaper import Config
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import datetime as dt
import csv
import numpy as np
import regex as re
# BDay is business day, not birthday...
from pandas.tseries.offsets import BDay
import requests
import json



user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36"
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10



def clean_text(text): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()) 



def score_flair(text):
  sentence = Sentence(text)
  classifier.predict(sentence)
  score = sentence.labels[0].score
  value = sentence.labels[0].value
  return score, value



# Function to get news headlines about 'search' on 'date'
def get_news(search, date):
    stories = []
    # timedelta for no. of days from 'date'
    delta = dt.timedelta(days = 1)
    
    # Instance of GoogleNews class in pygooglenews package
    gn = GoogleNews()
    
    # Search google for the news headlines
    result = gn.search(search, from_ = date.strftime('%m-%d-%Y'), to_ = (date+delta).strftime('%m-%d-%Y'))
        
    newsitem = result['entries']
    
    for item in newsitem:
        story = {
            'date':date,
            'title': item.title,
            'link': item.link,
            'published':item.published
        }
        stories.append(story)
    
    # Return all the headlines retrieved
    return stories
  
    
    

def get_gnews_article(df):
    list = []
    for i in range(0, df.shape[0]):
        dict = {}
        article = Article(df['link'][i], config = config)
        try:
            article.download()
            article.parse()
            article.nlp()       
            dict['Date'] = df['date'][i]
            dict['Title']=article.title
            dict['Article']=article.text
            dict['Summary'] = article.summary
            dict['Key_words'] = article.keywords
        except:
            pass
        
        list.append(dict)
        
    check_empty = not any(list)
        
    if check_empty == False:
        news_df = pd.DataFrame(list)
    else:
        news_df = pd.DataFrame()
    
    return news_df   




def calculate_sim_score(d1, d2):
    set_a, set_b = set(d1), set(d2)
    return len(set_a and set_b) / len(set_a or set_b)





def update_lexicon(sia):
    
    # stock market lexicon
    stock_lex = pd.read_csv('Resources/lexicon_data/stock_lex.csv')
    stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
    stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
    stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
    stock_lex_scaled = {}
    for k, v in stock_lex.items():
        if v > 0:
            stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
        else:
            stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

    # Loughran and McDonald
    positive = []
    with open('Resources/lexicon_data/LM_positive.csv', 'r') as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)
        for row in reader:
            positive.append(row[0].strip())
  
    negative = []
    with open('Resources/lexicon_data/LM_negative.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            entry = row[0].strip().split(" ")
            if len(entry) > 1:
                negative.extend(entry)
            else:
                negative.append(entry[0])

    final_lex = {}
    final_lex.update({word:2.0 for word in positive})
    final_lex.update({word:-2.0 for word in negative})
    final_lex.update(stock_lex_scaled)
    final_lex.update(sia.lexicon)
    sia.lexicon = final_lex

    


def make_sentences(text):
    """ Break apart text into a list of sentences """
    sentences = [sent for sent in split_single(text)]
    return sentences
    

    
    
def get_sentiments(news_df):
    
    nltk.download('vader_lexicon')
    nltk.download('punkt')

    sia = SentimentIntensityAnalyzer()
    
    update_lexicon(sia)        
        
    news_df_row = news_df.shape[0]
    news_df_col = news_df.shape[1]
    
    # Empty lists to append sentiment calculation
    news_sentiment_subjectivity_list = []
    news_sentiment_vader_comp_list = []

     #load the descriptions into textblob
    for i in range(news_df_row):
        desc_blob = [TextBlob(desc) for desc in (news_df.iloc[i, 1:news_df_col])]
                
        # Get subjectivity score for news_df
        news_sentiment_subjectivity_list.append([b.sentiment.subjectivity for b in desc_blob])
        
        # Get compound score from vader analysis
        news_sentiment_vader_comp_list.append([sia.polarity_scores(v)['compound'] for v in news_df.iloc[i, 1:news_df_col]])
    
    # Create dataframes from the generated lists
    news_subjectivity_df = pd.DataFrame(news_sentiment_subjectivity_list)
    news_vader_compound_df = pd.DataFrame(news_sentiment_vader_comp_list)
    
   
    news_sim_score = []

    #Calculate similarity score for weighing the sentiment score
    for i in range(news_df_row):
        sim_score_list = []
        for j in range(0, news_df_col-1):
            sim_scores = []
            for k in range(0,(news_df_col-1)):
                if j != k:
                    sim_scores.append(calculate_sim_score(news_df.iloc[i][j], news_df.iloc[i][k]))
            sim_score_list.append(np.mean(sim_scores))
        news_sim_score.append(sim_score_list)
    
    # Create the similarity score dataFrame    
    news_sim_score_df = pd.DataFrame(news_sim_score)
    
    # Sentiment score generated based on subjectivity score
    news_weighted_subj_senti_df = news_subjectivity_df * news_vader_compound_df
    
    #Sentiment score generated based on similarity score
    news_weighted_simi_senti_df = news_sim_score_df * news_vader_compound_df
    
    # Generate the sentiment dataframe
    news_sentiment_df = []
    news_sentiment_df = news_df[['index']]
    news_sentiment_df['subj_score'] = news_weighted_subj_senti_df.mean(axis = 1, skipna = True)
    news_sentiment_df['simi_score'] = news_weighted_simi_senti_df.mean(axis = 1, skipna = True)
    news_sentiment_df['vader_score'] = news_vader_compound_df.mean(axis = 1, skipna = True)
    
    return news_sentiment_df





def getTwitterData(hashtag, numDays):
        
    tweets_df = pd.DataFrame()
    
    today = dt.datetime.today()
    start = today - pd.Timedelta(days = numDays)
    
    #Since we are retrieving data we want to correlate with markets, we are using 'bdate_range' to only return business days                
    date_range = pd.bdate_range(end = dt.date.today(), periods = int(numDays))
       
    #Iterating through the date_range with an API call for each day to maximize data returned 
    # @TODO -- use function to limit API calls to not exceed limits
    for i in range(0, len(date_range) -1, 1):
    
        start_date = date_range[i]
        end_date = date_range[i+1] 
    
        start_date = pd.Timestamp.isoformat(start_date)
        end_date = pd.Timestamp.isoformat(end_date)
    

# Making the API call to retrieve tweets 

        source = "https://twitter32.p.rapidapi.com/getSearch"

# These parameters need to be set before running API call... could in the future receive inputs from user
## TODO -- if input is received, datetime will need to be formatted correctly for API call

        hashtag = hashtag

        querystring = {"hashtag": hashtag, "start_date": start_date,"end_date": end_date,"lang":"en"}
        headers = {
        'x-rapidapi-host': "twitter32.p.rapidapi.com",
        'x-rapidapi-key': "49e5cedc9fmsh6a2df83dacfc4c1p1c3469jsn3edb2e8cb9df"
        }
   
        
        response = requests.get(source, headers=headers, params=querystring).json()
        df = pd.DataFrame(response['data']['tweets']).T
        tweets_df = pd.concat([df, tweets_df])
        
    df_filtered = tweets_df.loc[:,["created_at","full_text", 'retweet_count']]
    df_filtered.index = pd.to_datetime(df_filtered['created_at'])
    df_filtered.index = df_filtered.index.date
    df_filtered.index.name = 'Date'
    df_filtered.drop(columns='created_at', inplace=True)
    df_filtered["full_text"] = df_filtered["full_text"].apply(clean_text)
    df_filtered = df_filtered.sort_index()
    
    return df_filtered


def get_flair_score(df):
    
    df_row = df.shape[0]
    df_col = df.shape[1]
    
    flair_list = []
    for i in range(df_row):
        flair_list.append((df.iloc[i, 1:df_col]).apply(lambda s: score_flair(s)[0]).mean())
        
    return flair_list