#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:32:34 2018

@author: travisbarton
"""

##Data preparation

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:25:46 2018

@author: travisbarton
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from random import choice, sample



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
    vec = vec.values
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
    isindex = np.where(noise.iloc[:,2] == 'physics')
    isntindex = np.where(data.iloc[:,2] != sub)
    isindex = sample(list(isindex[0]), 2000)
    isntindex = sample(list(isntindex[0]), 2000)
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
accs = []
##################################################################################
 
'''

1 sub


'''


subs = ['physics']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        


'''

2 subs

'''
subs = ['physics', 'bio']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        

'''
three subs

'''


subs = ['physics', 'bio', 'med']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        

'''

4 subs

'''


subs = ['physics', 'bio', 'med', 'geo']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        


'''

5 subs


'''

subs = ['physics', 'bio', 'med', 'geo', 'chem']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        


'''

6 subs


'''

subs = ['physics', 'bio', 'med', 'geo', 'chem', 'astro']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
noise = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/Training_data.csv")
noise.columns = data.columns
data = data.iloc[:,1:]
noise = noise.iloc[:,1:]
noise = noise.sample(4000)
#noise = Noise_maker(noise, 'physics')



Layer2_Spacy_vector = Turn_into_Spacy(data)
Layer1_Spacy_vector = Turn_into_Spacy(noise)
data.tag = Sub_treater(data.tag, subs)
noise.tag = Sub_treater(noise.tag, subs)


dat = np.empty([(data.shape[0]+noise.shape[0]), 301])
for i in range(data.shape[0]+noise.shape[0]):
    for j in range(300):
        if i < data.shape[0]:
            dat[i,j] = Layer2_Spacy_vector[i][j]
        else:
            dat[i,j] = Layer1_Spacy_vector[i - data.shape[0]][j]
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25)   

X_train = np.vstack([dat[data.shape[0]:,0:300], X_train])
y_train = np.concatenate([dat[data.shape[0]:, 300], y_train])



        
y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

model = Sequential()
model.add(Dense(50, input_dim = 300, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(len(subs)+1, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=200, batch_size=50, 
                          verbose = 0,
                          validation_data =[X_test[:,:300], y_test])


plt.figure()
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.subplot(212)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])


preds = model.predict(X_test[:,:300])


accs.append(Percent(y_test, preds))        




plt.figure()
plt.subplot(222)
plt.plot(accs)
plt.title("How do classes \n relate to accuracy")
plt.ylabel("Acc")
plt.xlabel("Number of classes")
plt.xticks([0, 1, 2, 3, 4, 5])
plt.yticks([.6, .65, .70, .75, .8, .85])
plt.savefig("class performance.png")
