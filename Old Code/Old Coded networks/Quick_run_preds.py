#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:10:58 2018

@author: travisbarton
"""

###Quick runs

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


def Turn_into_Spacy(data, tags = False):
    docs = []
    for i in range(data.shape[0]):
        temp  = (data.iloc[i, 1])
        if(str("'") in str(temp)):
            temp.replace(str("'"), "")
        if((temp[2] == '[' or temp[0] == '[')):
            for i in range(len(temp)):
                if(temp[i] == ']'):
                    temp = str(temp[i+1:])
                    break
        temp = temp[0:len(temp)-1]        
        docs.append(temp)
    textvectors = [[] for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        #print(docs[i])
        textvectors[i] = nlp(docs[i]).vector
       
    return(textvectors)
    

def grouper(vec):
    for i in range(len(vec)):
        if vec[i] == 'meta' or vec[i] != vec[i]:
            vec[i] = 'other'
    return(vec)

    
def Sub_treater(vec, sub):
    for i in range(len(vec)):
        if vec[i] not in sub:
            vec[i] = 'Not_{}'.format(sub)
    return(vec)

    
def Pred_to_num(pred):
    results = []
    for i in range(pred.shape[0]):
        results.append(int(max(np.where(pred[i,:] == max(pred[i,:])))))
    return(results)

    
def Percent(truth, test):
    correct = 0
    tr = Pred_to_num(truth)
    te = Pred_to_num(test)
    for i in range(truth.shape[0]):
        if tr[i] == te[i]:
            correct += 1
    return(correct/truth.shape[0])

    

def Noise_maker(data, sub):
    isindex = np.where(noise.iloc[:,2] == sub)
    isntindex = np.where(data.iloc[:,2] != sub)
    isindex = sample(list(isindex[0]), 100)
    isntindex = sample(list(isntindex[0]), 1000)
    index = [isindex, isntindex]
    final = pd.DataFrame(columns = ['id', 'Title', 'tag'])
    temp = [data.iloc[i,:] for i in isindex]
    temp2 = [data.iloc[i,:] for i in isntindex]
    k = 0
    for i in temp:
        final.loc[k,:] = i.values
        k = k+1
    for i in temp2:
        final.loc[k,:] = i.values
        k = k+1
    return(final)
    
  

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
nlp = spacy.load('en_vectors_web_lg')
RS = 100






#PHYSICS






data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]

Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['physics'])


dat = np.empty([(data.shape[0]), 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]

'''
FOR TEMP RUN
dat = easydat

'''
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   




        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Physics_Models/BestPhys.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)
model.load_weights("Physics_Models/BestPhys.hdf5")

physpreds = model.predict(X_test[:,:300])


''' 








bio















'''

data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]
Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['bio'])


dat = np.empty([data.shape[0], 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]

dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Bio_Models/BestBio.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)



model.load_weights("Bio_Models/BestBio.hdf5")

biopreds = model.predict(X_test[:,:300])












#MED










data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]

Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['med'])

dat = np.empty([data.shape[0], 301])

for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   

        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Med_Models/BestMed.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)




model.load_weights("Med_Models/BestMed.hdf5")

medpreds = model.predict(X_test[:,:300])







#GEO











data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]

Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['geo'])


dat = np.empty([(data.shape[0]), 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Geo_Models/BestGeo.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)



model.load_weights("Geo_Models/BestGeo.hdf5")

geopreds = model.predict(X_test[:,:300])











# CHEM









data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]

Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['chem'])


dat = np.empty([(data.shape[0]), 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Chem_Models/BestChem.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)



model.load_weights("Chem_Models/BestChem.hdf5")

chempreds = model.predict(X_test[:,:300])













# ASTRO








data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]


Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['astro'])




dat = np.empty([(data.shape[0]), 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        


X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   




        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  


y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   


model = Sequential()


model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Astro_Models/BestAstro.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)




model.load_weights("Astro_Models/BestAstro.hdf5")


astropreds = model.predict(X_test[:,:300])












# OTHER 








data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]

Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, ['astro', 'chem', 'geo', 'med', 'bio', 'physics'])
data.tag = Sub_treater_other(data.tag, ['astro', 'chem', 'geo', 'med', 'bio', 'physics'])



dat = np.empty([(data.shape[0]), 301])
for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]

onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   


        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(2, activation = 'softmax'))        
        
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
filepath="Other_Models/BestOther.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test]
                          , callbacks=callbacks_list)


model.load_weights("Other_Models/BestOther.hdf5")

otherpreds = model.predict(X_test[:,:300])





















