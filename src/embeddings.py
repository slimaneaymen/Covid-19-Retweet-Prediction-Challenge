from numpy import array
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


def mean_emb(df):
    v1 = np.mean(df['v1'].tolist())
    v2 = np.mean(df['v2'].tolist())
    v3 = np.mean(df['v3'].tolist())
    v4 = np.mean(df['v4'].tolist())
    v5 = np.mean(df['v5'].tolist())
    return pd.DataFrame([[v1,v2,v3,v4,v5]], columns = ['v1', 'v2', 'v3', 'v4', 'v5'])

def encodeWords(imp_words):
    #Encoding important words
    encodings = pd.DataFrame(imp_words, columns=['word'])
    encodings['code'] = range(1, len(encodings) + 1)
    vocab = len(imp_words)+1
    return encodings, vocab

def embedSentences(text, encodings, imp_words):
    #Removing non-important words from tweets
    embedded_sentences = []
    for txt in text:
        tokenized = word_tokenize(txt)
        simple_L = [w for w in tokenized if w in imp_words]
        #Replace words with their encodings
        for i, word in enumerate(simple_L):
            simple_L[i] = encodings.loc[encodings['word'] == word, 'code'].values[0]
        embedded_sentences.append(simple_L)
    return embedded_sentences


def createEmbeddings(embedded_sentences, encodings, vocab, labels, emb_size=5, epochs=5):
    #Find maximum number of words
    word_count = lambda text: len(text)
    longest_sentence = max(embedded_sentences, key=word_count)
    length_long_sentence = len(longest_sentence)

    #Pad Sentences
    padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')

    #Build model
    model = Sequential()
    model.add(Embedding(vocab, emb_size, input_length=length_long_sentence))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    #Train model
    model.fit(padded_sentences, labels, epochs=epochs, verbose=0)
    loss, accuracy = model.evaluate(padded_sentences, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    #Get Embeddings
    embeddings = model.layers[0].get_weights()[0]
    embeddings = pd.DataFrame(embeddings)
    embeddings.columns = ['v1', 'v2', 'v3', 'v4', 'v5']
    embeddings['word'] = encodings['word']

    return embeddings, loss, accuracy

    
def accumulateEmbeddings(embedded_sentences, embeddings):
    #Empty Dataframe
    emptydf = pd.DataFrame([[0,0,0,0,0]], columns = ['v1', 'v2', 'v3', 'v4', 'v5'])

    acc_emb = []
    for text in embedded_sentences:
        acc = pd.DataFrame()
        for w in text:
            emb = embeddings.iloc[w-1] #Series
            emb.drop(['word'], inplace=True)
            acc = acc.append(emb, ignore_index=True)
        if len(acc) < 1 :
            acc_emb.append(emptydf) #append empty dataframe
        elif len(acc) > 1 :
            acc_emb.append(mean_emb(acc)) #find mean
        else:
            acc_emb.append(acc) #append single dataframe

    acc_emb = pd.concat(acc_emb)
    return acc_emb

