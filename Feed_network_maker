#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:15:11 2018

@author: travisbarton
"""
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from random import choice, sample
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

nlp = spacy.load('en_vectors_web_lg')
RS = 69




def Sub_treater(vec, sub):
    for i in range(len(vec)):
        if vec[i] not in sub:
            vec[i] = 'Not_{}'.format(sub)
    return(vec)


def Binary_network(X, Y, val_split, nodes, epochs, batch_size, label):
    model = Sequential()

    model.add(Dense(nodes, input_dim = X.shape[1], activation = 'linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha = .001))
    model.add(Dense(2, activation = 'softmax'))        
            
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    #filepath="Best_{}.hdf5".format(label)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
     #                            save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    model_history = model.fit(X_train[:,:300], y_train, 
                              epochs=epochs, batch_size=batch_size, 
                              verbose = 0, validation_split = val_split)
    return(model.predict(X)[:,0])

def Feed_reduction(vectors, labels, val_split, nodes, epochs, batch_size):
    temp_vec = vectors.copy()
    temp_lab
    
    final_results = np.empty([vectors.shape[0], len(np.unique(labels))])
    for label in np.unique(labels):
        temp_lab = Sub_treater(temp_lab, label)
        
        




########### Testing #################
        
easydata = pd.read_csv("/Users/travisbarton/Documents/GitHub/Redditbot/VDifferentData.csv")
easydata.columns = ['id', 'title', 'tag']

easydat = np.empty([easydata.shape[0],301])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
easydat[:,300] = easydata.tag 

dat = easydat.copy()

for i in range(dat.shape[0]):
    if dat[i,300] == 1:
        pass
    else:
        dat[i,300] = 0

onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.15, 
                                                    random_state=RS)   


y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

temp = Binary_network(X_train, y_train, .1, 50, 15, 30, "dont matter yet")

temp = np.round(temp).astype(int)


Pred_to_num(y_train)[0:10]

1-sum(temp == Pred_to_num(y_train))/dat.shape[0]
#When you pick up next time. You are working on integrating the binary networks
#you have it returning the probability of being in the first column rn.
# next time erase label paremeter and input x_test parameter























