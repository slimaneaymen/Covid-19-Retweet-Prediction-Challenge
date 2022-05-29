import csv
import numpy as np
import pandas as pd

def new_word(word_dict, word, classes):
    new_word = dict(count=0)
    for c in range(classes):
        new_word["count_"+str(c)] = 0
        new_word["ig_"+str(c)] = 0
    word_dict[word] = new_word

def entropy(probs, pr=False):
    entr = 0
    for p in probs:
        if p == 1: return 0
        if p != 0:
            entr -= (p * np.log2(p))
            if pr: print("entr so far ", entr)
    return entr

def ig(text, y, classes): #text splitted
    n = text.size #text is 'Series'
    print("Number of tweets: ", n)
    
    words = dict() #dictionary
    seen = set() #set
    for i in range(n):
        for w in text[i]:
            if w not in seen:
                seen.add(w)
                new_word(words, w, classes)
            words[w]['count'] += 1
            y_bin = str(y.at[i,'bin']) #get true bin
            words[w]['count_'+y_bin] +=1
    
    #then calculate probabilities of bins
    groups = y.groupby(['bin']).size().reset_index(name='count')
    groups['prob'] = groups['count'] / n
    print(groups)
    #entropy calculation
    probs = []
    for index, group in groups.iterrows():
        probs.append(group[2])
    bins_entr = entropy(probs, False)
    print("Bins entropy: ", bins_entr)

    for w in words:
        for c in range(classes):
            prob = words[w]['count_'+str(c)] / words[w]['count']
            probs = []
            probs.append(prob)
            probs.append(1-prob)
            entr = entropy(probs)
            words[w]['ig_'+str(c)] = bins_entr - entr
    return words

def information_gain(text, y_train, n):
    bins = [-1,0,1000,10000,100000,1000000]
    vocab = 5*n + 1

    #Split data into bins
    y = pd.DataFrame({"retweets":y_train})
    y["bin"] = pd.cut(x=y_train,
                      bins=bins,
                      labels=range(0,5) )
    print("Splitted in bins", len(bins)-1)

    #Split text into words
    txt = pd.Series(text)
    txt = txt.str.split(" ")
    words = ig(txt, y, 5)
    
    #Sort
    words0 = sorted(words.items(), key=lambda x: x[1]['ig_0'], reverse=True)
    words0 = [w for w in words0 if w[1]['count_0'] != 0 and w[1]['count'] > 10]
    
    words1 = sorted(words.items(), key=lambda x: x[1]['ig_1'], reverse=True)
    words1 = [w for w in words1 if w[1]['count_1'] != 0 and w[1]['count'] > 10]
    
    words2 = sorted(words.items(), key=lambda x: x[1]['ig_2'], reverse=True)
    words2 = [w for w in words2 if w[1]['count_2'] != 0 and w[1]['count'] > 10]
    
    words3 = sorted(words.items(), key=lambda x: x[1]['ig_3'], reverse=True)
    words3 = [w for w in words3 if w[1]['count_3'] != 0 and w[1]['count'] > 10]
    
    words4 = sorted(words.items(), key=lambda x: x[1]['ig_4'], reverse=True)
    words4 = [w for w in words4 if w[1]['count_4'] != 0 and w[1]['count'] > 10]

    return words0[:n], words1[:n], words2[:n], words3[:n], words4[:n], y, vocab
    
