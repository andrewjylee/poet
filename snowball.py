import os
import codecs
import re
import nltk
import numpy as np

def get_init_words(model, length):
    return [i for i in model.keys() if len(i) == length]

def get_next_words(model, prev):
    return model[prev].keys()



#with open('./data/partial_various_poets') as f:
with codecs.open('./data/partial_various_poets') as f:
    data = f.read()

    data = data.replace("\n", " ")
    data = data.replace(".", " ")
    data = data.replace(",", " ")
    data = data.replace("\r", " ")
    data = data.lower().replace(";", " ")
    words = nltk.wordpunct_tokenize(re.sub('[^a-zA-Z_ ]', '', data))

    model = {}
    inverse_model = {}

    for i in range(len(words) - 1):
        if len(words[i]) + 1 == len(words[i+1]):
            #print words[i], words[i+1]
            if words[i] not in model.keys():
                model[words[i]] = {}

            if words[i+1] not in model[words[i]].keys():
                model[words[i]][words[i+1]] = 0

            model[words[i]][words[i+1]] += 1
            
        if len(words[i]) - 1 == len(words[i+1]):
            if words[i] not in inverse_model.keys():
                inverse_model[words[i]] = {}

            if words[i+1] not in inverse_model[words[i]].keys():
                inverse_model[words[i]][words[i+1]] = 0

            inverse_model[words[i]][words[i+1]] += 1

    print model

    for _, k in model.iteritems():
        #print _, k
        summation = sum([i for i in k.values()])
        for i, v in k.iteritems():
            k[i] = float(v) / summation

    for _, k in inverse_model.iteritems():
        #print _, k
        summation = sum([i for i in k.values()])
        for i, v in k.iteritems():
            k[i] = float(v) / summation

    for i in range(50):
        next_word = np.random.choice(get_init_words(model, 2), size = 1, p = None)[0]
        print '--'
        for _ in range(10):
            print next_word
            if next_word not in model.keys():
                break
            next_word = np.random.choice(model[next_word].keys(), size = 1, p = model[next_word].values())[0]
