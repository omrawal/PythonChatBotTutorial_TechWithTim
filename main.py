# https://www.techwithtim.net/tutorials/ai-chatbot/part-1/


import json
import pickle
import random
import tensorflow
from tensorflow.python.framework import ops
import tflearn
import numpy
import codecs
import nltk
from nltk import tag
from nltk.stem.lancaster import LancasterStemmer
from numpy.core.defchararray import mod
from tflearn.activations import softmax
from tflearn.layers.core import activation
stemmer = LancasterStemmer()

file = open("intents.json")
data = json.load(file)

try:
    f = open("data.pickle", 'rb')
    words, lables, training, output = pickle.load(f)
except:
    words = []
    lables = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            if(intent['tag']not in lables):
                lables.append(intent['tag'])

    words = [stemmer.stem(w.lower())for w in words if w != "?"]
    words = sorted(list(set(words)))
    lables = sorted(lables)

    training = []
    output = []
    out_empty = [0 for _ in range(len(lables))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[lables.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    f = open("data.pickle", 'wb')
    pickle.dump((words, lables, training, output), f)


ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
