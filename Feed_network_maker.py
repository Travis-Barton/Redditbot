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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def Sub_treater(vec, sub):
    holder = []
    for i in range(len(vec)):
        if vec[i]!= sub:
            holder.append('Not_{}'.format(sub))
        else:
            holder.append(sub)
    return(holder)


def Binary_network(X, Y, X_test, label, val_split, nodes, epochs, batch_size):
    model = Sequential()

    model.add(Dense(nodes, input_dim = X.shape[1], activation = 'linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.4))
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
    model_history = model.fit(X[:,:300], Y, 
                              epochs=epochs, batch_size=batch_size, 
                              verbose = 0, validation_split = val_split)
    return([model.predict(X)[:,0], model.predict(X_test)[:,0]])

def Feed_reduction(X, Y, X_test, labels, val_split = .1, nodes = None, epochs = 15, batch_size = 30):
    if nodes == None:
        nodes = np.round(X.shape[0]/4)
    onehot_encoder = OneHotEncoder(sparse=False) 
    finaltrain = np.empty([X.shape[0], len(labels)])
    finaltest = np.empty([X_test.shape[0], len(labels)])
    i = 0
    for label in labels:
        x = X.copy()
        y = Y.copy()
        x_test = X_test.copy()
        y = Sub_treater(y, str(label))
        y = pd.factorize(y)[0]
        y = y.reshape(len(y), 1).astype(int)        
        y = onehot_encoder.fit_transform(y)
        temp = Binary_network(x, y, x_test, label, val_split, nodes, epochs, batch_size)
        finaltrain[:,i] = temp[0]
        finaltest[:,i] = temp[1]
    return([finaltrain, finaltest])
        




########### Testing #################
        
easydata = pd.read_csv("/Users/travisbarton/Documents/GitHub/Redditbot/VDifferentData.csv")
easydata = pd.read_csv(r"C:\Users\sivar\OneDrive\Documents\GitHub\Redditbot\VDifferentData.csv")
easydata.columns = ['id', 'title', 'tag']

easydat = np.empty([easydata.shape[0],301])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
tags = easydata.tag 
easydat[:,300] =easydata.tag
dat = easydat.copy()


        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.15, 
                                                    random_state=RS)   


y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

temp = Binary_network(X_train, y_train, X_test, "dont matter yet", .1, 50, 15, 30)


train_res = np.round(temp[0]).astype(int)
test_res = np.round(temp[1]).astype(int)


Pred_to_num(y_test)[0:10]

1-sum(train_res == Pred_to_num(y_train))/dat.shape[0]
1-sum(test_res == Pred_to_num(y_test))/dat.shape[0]
plot_confusion_matrix(confusion_matrix(Pred_to_num(y_train), train_res), [0,1], normalize = True, title = "Is test good?")
plot_confusion_matrix(confusion_matrix(Pred_to_num(y_test), test_res), [0,1], normalize = True, title = "Is test good?")


#When you pick up next time. You are working on integrating the binary networks
#you have it returning the probability of being in the first column rn.
# next time erase label paremeter and input x_test parameter



############### test #2 
easydat = np.empty([easydata.shape[0],300])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
tags = easydata.tag 

tag = []
for i in range(len(tags)):
    if tags.iloc[i] == 1.0:
        tag.append('Aww')
    elif tags.iloc[i] == 2.0:
        tag.append('Politics')
    else:
        tag.append("ST")

dat = easydat.copy()

onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    tag, 
                                                    test_size=0.25, 
                                                    random_state=RS)   








results = Feed_reduction(X_train, y_train, X_test, np.unique(y_train), nodes = 50)
new_X = results[0]
new_X_test = results[1]

''' last you left off, it works, but is returning CRAZY high numbers... weird'''










