#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:49:33 2018

@author: travisbarton
"""

#Easy data




import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from collections import Counter
import datetime
import time
import requests

#Setup... This contains passwords and access to my reddit account, So I will not be sharing the data unedited. 

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username=base64.b64decode('bWF0aF9pc19teV9yZWxpZ2lvbg=='), 
                     password=(base64.b64decode("U2lyemlwcHkx")))
ST = reddit.subreddit('showerthoughts')

easydata = pd.read_csv("/Users/travisbarton/Documents/GitHub/Redditbot/VDifferentData.csv")
easydata.columns = ['id', 'title', 'tag']


i = easydata.shape[0]

for post in ST.top("all", limit = 1000):
    easydata.loc[i,:] = [i, post.title, 3]
    i = i+1
    
    
   
    
    
    
### RUN BELOW
easydat = np.empty([easydata.shape[0],301])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
easydat[:,300] = easydata.tag 

dat = easydat.copy()

onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   




        


'''
svm

'''

clf = svm.SVC(gamma = 'scale', kernel = 'linear')
clf.fit(X_train, y_train)
svmpreds = clf.predict(X_test)
len(np.where(svmpreds == y_test)[0])/len(y_test)


confm = confusion_matrix((y_test), (svmpreds))
confm
plot_confusion_matrix(confm, ['aww', 'politics', 'showerthoughts'], 
                      normalize = True, title = "Which subreddit?")




'''

Full Network


'''



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
model.add(Dense(3, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=50, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test])

easypreds = model.predict(X_test[:,:300])

plt.figure()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title("Accuracy")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(212)
plt.title("\nLoss")
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
   

confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(easypreds))
confm
plot_confusion_matrix(confm, ['aww', 'politics', 'showerthoughts'], 
                      normalize = True, title = "Which subreddit?")




'''


Series of networks



'''

#1


dat = easydat.copy()
for i in range(dat.shape[0]):
    if dat[i, 300] != 1:
        dat[i, 300] = 2



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

#model.load_weights("Physics_Models/weights-improvement-148-0.86.hdf5")
model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test])

easypreds1 = model.predict(X_test[:,:300])
easy1 = model.predict(X_train)

confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(easypreds1))
confm
plot_confusion_matrix(confm, ['aww', 'other'], 
                      normalize = True, title = "Which subreddit?")




#2

dat = easydat.copy()
for i in range(dat.shape[0]):
    if dat[i, 300] != 2:
        dat[i, 300] = 2
    else:
        dat[i, 300] = 1



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

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test])

easypreds2 = model.predict(X_test[:,:300])
easy2 = model.predict(X_train)

confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(easypreds2))
confm
plot_confusion_matrix(confm, ['Politics', 'other'], 
                      normalize = True, title = "Which subreddit?")




#3
dat = easydat.copy()
for i in range(dat.shape[0]):
    if dat[i, 300] != 3:
        dat[i, 300] = 2
    else:
        dat[i,300] = 1



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

model_history = model.fit(X_train[:,:300], y_train, epochs=20, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test[:,:300], y_test])

easypreds3 = model.predict(X_test[:,:300])
easy3 = model.predict(X_train)

confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(easypreds3))
confm
plot_confusion_matrix(confm, ['Shower \n Thoughts', 'other'], 
                      normalize = True, title = "Which subreddit?")





#### Put it all together


X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   


Final_x_train = np.vstack([easy1[:,1], easy2[:,0], easy3[:,0]]).T
Final_y_train = y_train

Final_x_test = np.vstack([easypreds1[:,1], easypreds2[:,0], easypreds3[:,0]]).T
Final_y_test = y_test


        
Final_y_train = Final_y_train.reshape(len(Final_y_train), 1).astype(int)
Final_y_test = Final_y_test.reshape(len(Final_y_test), 1).astype(int)  

Final_y_train = onehot_encoder.fit_transform(Final_y_train)
Final_y_test = onehot_encoder.fit_transform(Final_y_test)   


model = Sequential()
model.add(Dense(12, input_dim = 3, activation = 'sigmoid'))
model.add(Dense(13, activation = 'sigmoid'))
#model.add(Dropout(.4))
model.add(Dense(15, activation = 'sigmoid'))
#model.add(Dropout(.5))
model.add(Dense(3, activation = 'softmax'))


model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', 
              metrics=['accuracy'])



model_history = model.fit(vvv[:,0:3], vvv[:,3:], epochs=500, batch_size=50, 
                          verbose = 1,
                          validation_data =[Final_x_test, Final_y_test])

plt.figure()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title("Accuracy")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(212)
plt.title("\nLoss")
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])

plt.savefig("Toy Final Performance.png")




Fullpreds = model.predict(Final_x_test)


Percent(Final_y_test, Fullpreds)        
confm = confusion_matrix(Pred_to_num(Final_y_test), Pred_to_num(Fullpreds))
confm
plot_confusion_matrix(confm, np.hstack(['aww', 'politics', 'showerthoughts']), 
                      normalize = True, title = "Which subreddit?")



'''


SVM


'''



clf = svm.SVC(gamma = 'scale', kernel = 'linear', degree = 3)
clf.fit(Final_x_train, y_train)
svmpreds = clf.predict(Final_x_test)
len(np.where(svmpreds == y_test)[0])/len(y_test)
confm = confusion_matrix(Pred_to_num(Final_y_test), (svmpreds-1))
plot_confusion_matrix(confm, np.hstack(['aww', 'politics', 'showerthoughts']), 
                      normalize = True, title = "Which subreddit?")








    