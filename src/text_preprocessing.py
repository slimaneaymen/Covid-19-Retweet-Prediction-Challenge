import csv
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict
import re

from sklearn.feature_extraction.text import CountVectorizer

from ig_testing import information_gain

###Installing required packages
###Uncomment to download
##nltk.download('stopwords')
##nltk.download('punkt')
##nltk.download('wordnet')
##nltk.download('averaged_perceptron_tagger')

def text_treatement(text):
  #Convert string to lower or upper case
  text = text.lower() 
  #Remove Special Character
  text=re.sub('[^A-Za-z0-9]+', ' ', text)
  #Stop word removal
  en_stops = set(stopwords.words('english')+ stopwords.words('french')+stopwords.words('german'))
  
  tokenized = word_tokenize(text)
  text = " ".join([word for word in tokenized if (word not in en_stops) and not(word.isnumeric() and word!='19')])
  
  #POS tag & lamenation
  tag_map = defaultdict(lambda : wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV
  
  lemma=[]
  text_word_tokens = word_tokenize(text)
  lemmatizer = WordNetLemmatizer()
  pos = pos_tag(text_word_tokens)
  for token, tag in pos:
    lemma.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))
  lemma=" ".join([word for word in lemma])
  return lemma

