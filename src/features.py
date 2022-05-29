import pandas as pd
import numpy as np

def new_word(word_dict, word):
    new_word = dict()
    new_word['count'] = 0
    new_word['idf'] = 0
    word_dict[word] = new_word

def feat_test(text, words, train=False):
    splitted_text = text.str.split(",").tolist()
    n = len(splitted_text)
    
    count = []
    idf = []
    #Calculate minimum idf and count
    for i in splitted_text:
        num = len(i)
        if(num==1) and (not i[0]): #check if string is empty
            num = 0
        count.append(num)
        if(num==0): #if there is no feature idf
            idf.append(0)
            continue
        freq=[]
        for w in i:
            df = words[w.strip()]['count']
            if(train):
                words[w.strip()]['idf'] = np.log(n/(df+1))
            freq.append( words[w.strip()]['idf'] )
                
        idf.append(np.min(freq)) #minimum idf
    return count, idf

def feat_treatment(feature):
    splitted = feature.str.split(",").tolist() #list of lists
    n = len(splitted)
    
    words = dict()
    seen = set()
    for i in splitted:
        for w in i:
            word = w.strip() #trim
            if word not in seen: #found new word
                seen.add(word)
                new_word(words,word)
            words[word]['count'] += 1 #add 1
    count, idf = feat_test(feature, words, train=True)
    return count, idf, words

def boolToNum(booleans):
    booleans = booleans.to_numpy()
    booleans = booleans.astype(int)
    booleans = pd.Series(booleans)
    return booleans
