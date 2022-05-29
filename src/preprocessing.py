import csv
import time
import pandas as pd
import numpy as np

from ig_testing import information_gain
from text_preprocessing import text_treatement
from hashtags import hashtag_treatment
from features import feat_treatment
from features import boolToNum
from timestamp import timestamp_treatment
from embeddings import encodeWords
from embeddings import embedSentences
from embeddings import createEmbeddings
from embeddings import accumulateEmbeddings

def preprocessing_train(X_train, Y_train, fileName):
  start = time.time()

  #User Timestamp Feature
  print("Timestamp Feature")
  timestamp = timestamp_treatment(X_train['timestamp'])
  monthday = dict()
  for i in range(len(timestamp)):
    monthday[i] = (round(timestamp[i]['md_idf'], 3)) #3point float
  monthday = pd.Series(monthday)

  #Verified User Feature
  print('User_verified features')
  verified = X_train['user_verified'].to_numpy()
  verified = verified.astype(int)
  verified = pd.Series(verified)

  #Hashtag Features
  print("Hashtag features")
  n_hash, freq_hash, hashtags = hashtag_treatment(X_train['hashtags'])
  n_hash = pd.Series(n_hash)
  freq_hash = pd.Series(freq_hash)

  #Url Features
  print("Url features")
  n_url, freq_url, urls = feat_treatment(X_train['urls'])
  n_url = pd.Series(n_url)
  freq_url = pd.Series(freq_url)

  #User Mentions Features
  print("Mentions features")
  n_tag, freq_tag, tags = feat_treatment(X_train['user_mentions'])
  n_tag = pd.Series(n_tag)
  freq_tag = pd.Series(freq_tag)

  #Preprocessing text
  print("Preprocesssing text")
  L=[]  
  for i in X_train.text:
    L.append(text_treatement(i))

  #Text Features
  print("Calculating Information Gain")
  words0,words1,words2,words3,words4, ybins, vocab = information_gain(L, Y_train, 10)

  print("\nVocabulary:", vocab)

  imp_words = []
  for i in range(10):
      imp_words.append(words0[i][0])
      imp_words.append(words1[i][0])
      imp_words.append(words2[i][0])
      imp_words.append(words3[i][0])
      imp_words.append(words4[i][0])
  print(imp_words)

  labels = ybins['bin'].to_numpy()


  print("\nEncode important words")
  encodings, vocab = encodeWords(imp_words)
  print("Embed train preprocessed sentences")
  embedded_sentences = embedSentences(L, encodings, imp_words) #embed train preprocessed sentences
  print("Calculating embeddings")
  word_embeddings, loss, accuracy = createEmbeddings(embedded_sentences, encodings, vocab, labels)
  print("\nAccumulate embeddings")
  embeddings = accumulateEmbeddings(embedded_sentences, word_embeddings) #accumulate train embeddings

  embeddings.columns = ['v1','v2','v3','v4','v5']

  #Create new csv
  print("\nAppending preprocessed features")
  new_train = X_train[['id','user_statuses_count','user_followers_count','user_friends_count']]
  #Add timestamp feature
  new_train.insert(1, 'timestamp_freq', monthday.values)
  #Add user_verified feature
  new_train.insert(2,'user_verified', verified)
  #Add mentions features
  new_train = new_train.assign(mentions_count=n_tag.values)
  new_train = new_train.assign(mentions_freq=freq_tag.values)
  #Add urls features
  new_train = new_train.assign(urls_count=n_url.values)
  new_train = new_train.assign(urls_freq=freq_url.values)
  #Add hashtags features
  new_train = new_train.assign(hashtags_count=n_hash.values)
  new_train = new_train.assign(hashtags_freq=freq_hash.values)
  #Add text features
  new_train_plusemb = new_train.assign(v1=embeddings['v1'].values) #v1
  new_train_plusemb = new_train_plusemb.assign(v2=embeddings['v2'].values) #v2
  new_train_plusemb = new_train_plusemb.assign(v3=embeddings['v3'].values) #v3
  new_train_plusemb = new_train_plusemb.assign(v4=embeddings['v4'].values) #v4
  new_train_plusemb = new_train_plusemb.assign(v5=embeddings['v5'].values) #v5

  #new_train_plusemb = pd.concat([new_train, text_emb], axis=1)

  #Write to file
  print("\nWriting new csv")
##  new_train.to_csv('without_emb.csv')
  new_train_plusemb.to_csv(fileName, index=False)

  end = time.time()
  print("\nTime passed")
  print((end-start)/60)

  return imp_words, encodings, word_embeddings

def preprocessing_test(X_test, imp_words, encodings, word_embeddings, fileName):
  start = time.time()

  #Reset index
  X_test.reset_index(drop=True, inplace=True)

  #User Timestamp Feature
  print("Timestamp Feature")
  timestamp = timestamp_treatment(X_test['timestamp'])
  monthday = dict()
  for i in range(len(timestamp)):
    monthday[i] = (round(timestamp[i]['md_idf'], 3)) #3point float
  monthday = pd.Series(monthday)

  #Verified User Feature
  print('User_verified features')    
  #verified = boolToNum(X_test['user_verified'])
  verified = X_test['user_verified'].to_numpy()
  verified = verified.astype(int)
  verified = pd.Series(verified)

  #Hashtag Features
  print("Hashtag features")
  # n_hash, freq_hash = hashtag_test(X_test['hashtags'], hashtags)
  n_hash, freq_hash, h = hashtag_treatment(X_test['hashtags'])
  n_hash = pd.Series(n_hash)
  freq_hash = pd.Series(freq_hash)

  #Url Features
  print("Url features")
  # n_url, freq_url= feat_test(X_test['urls'], urls)
  n_url, freq_url, u= feat_treatment(X_test['urls'])
  n_url = pd.Series(n_url)
  freq_url = pd.Series(freq_url)

  #User Mentions Features
  print("Mentions features")
  # n_tag, freq_tag = feat_test(X_test['user_mentions'], tags)
  n_tag, freq_tag, t= feat_treatment(X_test['user_mentions'])
  n_tag = pd.Series(n_tag)
  freq_tag = pd.Series(freq_tag)

  #Preprocessing text
  print("Preprocesssing text")
  L=[]  
  for i in X_test.text:
      L.append(text_treatement(i))

  print("Embed train preprocessed sentences")
  embedded_sentences = embedSentences(L, encodings, imp_words) #embed test preprocessed sentences
  print("Accumulate embeddings")
  embeddings = accumulateEmbeddings(embedded_sentences, word_embeddings) #accumulate test embeddings

  embeddings.columns = ['v1','v2','v3','v4','v5']

  #Create new csv
  print("\nAppending preprocessed features")
  new_test = X_test[['id','user_statuses_count','user_followers_count','user_friends_count']]
  #Add timestamp feature
  new_test.insert(1, 'timestamp_freq', monthday.values)
  #Add user_verified feature
  new_test.insert(2,'user_verified', verified)
  #Add mentions features
  new_test = new_test.assign(mentions_count=n_tag.values)
  new_test = new_test.assign(mentions_freq=freq_tag.values)
  #Add urls features
  new_test = new_test.assign(urls_count=n_url.values)
  new_test = new_test.assign(urls_freq=freq_url.values)
  #Add hashtags features
  new_test = new_test.assign(hashtags_count=n_hash.values)
  new_test = new_test.assign(hashtags_freq=freq_hash.values)
  #Add text features
  new_test_plusemb = new_test.assign(v1=embeddings['v1'].values) #v1
  new_test_plusemb = new_test_plusemb.assign(v2=embeddings['v2'].values) #v2
  new_test_plusemb = new_test_plusemb.assign(v3=embeddings['v3'].values) #v3
  new_test_plusemb = new_test_plusemb.assign(v4=embeddings['v4'].values) #v4
  new_test_plusemb = new_test_plusemb.assign(v5=embeddings['v5'].values) #v5

  #Write to file
  print("\nWriting new csv")
  new_test_plusemb.to_csv(fileName, index=False)

   
  end = time.time()
  print("\nTime passed")
  print((end-start)/60)

