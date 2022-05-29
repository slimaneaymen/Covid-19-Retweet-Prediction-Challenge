from datetime import datetime
import csv
import time
import numpy as np
import pandas as pd


def new_m_d(tmp_dict,tmp):
    new_tmp = dict()
    new_tmp["m_d_frequency"] = 0
    tmp_dict[tmp] = new_tmp
    
def new_time(time_dict,time,i):
    new_time = dict()
    new_time["m_d"] = time[5:13]
    time_dict[i]= new_time
def new_tweet(tweet_dict,tweet):
    new_tweet = dict()
    new_tweet["md_idf"] = 0    
    tweet_dict[tweet]= new_tweet
    
def timestamp_treatment(time): 
    n = len(time)     
    tmp = dict()
    t=dict()
    t_md= dict()
    seen_md = set() #set
    
    for i in range(n):
        l=datetime.utcfromtimestamp(time[i]/1000).strftime("%Y-%m-%d %H:%M:%S")
        new_time(tmp,l,i)
        if l[5:13] not in seen_md:
            seen_md.add(l[5:13])
            new_m_d(t_md,l[5:13])
        t_md[l[5:13]]['m_d_frequency'] += 1
        
    for i in range(n):
        Q_idf_md=dict()

        for w in tmp[i]:
            new_tweet(t,i)            
            df=t_md[tmp[i]['m_d']]['m_d_frequency']
            idf=np.log(n/(df+1))
            t[i]["md_idf"] = idf            
            
    return t
