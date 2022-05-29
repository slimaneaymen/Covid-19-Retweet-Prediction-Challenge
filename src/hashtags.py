from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np

def new_word(word_dict, word):
    new_word = dict()
    new_word["Doc_frequency"] = 0
    new_word["idf"] = 0
    word_dict[word] = new_word
def new_tweet(tweet_dict,tweet):
    new_tweet = dict()
    new_tweet["tf_idf"] = 0
    new_tweet["idf"] = 0    
    new_tweet["number of #"] = 0
    tweet_dict[tweet]= new_tweet
    
def check_hash(word):
    a=word
    a=list(a)
    j=0
    for i in a:
        if i=='#':
            j=j+1
    return j       

def frequency(text): #text splitted
    n = len(text) #text is 'Series'    
    words = dict() #dictionary
    tweets= dict()
    seen = set() #set
    for i in range(n):
        if isinstance(text[i], float)==False:
            seen_t=[]
            for w in text[i]:
                if w.lower().strip() not in seen:
                        seen.add(w.lower().strip())
                        new_word(words, w.lower().strip())
                words[w.lower().strip()]['Doc_frequency'] += 1
    for i in range(n):
        Q_idf=dict()
        Q_tf_idf=dict()
        if isinstance(text[i], float)==False:
            y=[x.lower().strip() for x in text[i]]
            words_count = Counter(y)        
            for w in y:
                tf=words_count[w]/len(text[i])
                df=words[w]['Doc_frequency']
                idf=np.log(n/(df+1))
                words[w]['idf'] = idf
                Q_idf[w]=idf
                Q_tf_idf[w]=tf*idf
        
            new_tweet(tweets,i)
            tweets[i]["tf_idf"] = Q_tf_idf
            tweets[i]["idf"] = Q_idf            
            if len(text[i])==1 and not text[i][0]: #check empty string
                tweets[i]['number of #'] = 0
            else:
                tweets[i]["number of #"] = len(text[i])

        else:       
            new_tweet(tweets,i)
  
    return words,tweets


def hashtag_treatment(hashtags):
    n = len(hashtags)
    txt = pd.Series(hashtags)
    txt = list(txt.str.split(","))

    words1, tweets1 = frequency(txt)
    count = []
    freq = []
    for key, value in tweets1.items(): #key->tweet number value->dictionary
        tf_idf = value['tf_idf'] #dictionary
        idf_dict = value['idf'] #dictionary or 0
        hashs = value['number of #'] #int
        count.append(hashs)
        if hashs==0: #isinstance(idf_dict, int)
            freq.append(0)
        else:
            idf = list(idf_dict.values()) #list of values
            freq.append(np.min(idf))
    return count, freq, words1
    
