from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')
# Import flair Sentence to process input text
from flair.data import Sentence
import regex as re


def clean_text(text): 
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()) 

def score_flair(text):
  sentence = Sentence(text)
  classifier.predict(sentence)
  score = sentence.labels[0].score
  value = sentence.labels[0].value
  return score, value