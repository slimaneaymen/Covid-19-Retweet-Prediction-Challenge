import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import preprocessing_train
from preprocessing import preprocessing_test

#Read train data
train_data = pd.read_csv("data/train.csv", keep_default_na=False)

eval_data = pd.read_csv("data/evaluation.csv", keep_default_na=False)

Y_train = train_data['retweet_count']
X_train = train_data.drop(['retweet_count'], axis=1)

#Preprocessing train data
imp_words, encodings, word_embeddings = preprocessing_train(X_train, Y_train, 'preprocessed_train.csv')

#Preprocessing test data
preprocessing_test(eval_data, imp_words, encodings, word_embeddings, 'preprocessed_eval.csv')

