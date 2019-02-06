#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:09:52 2018

@author: travisbarton
"""
# This is a test




import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from collections import Counter
import datetime
import time
import requests
from Feed_network_maker import Sub_treater, Binary_network, Feed_reduction
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

#Setup... This contains passwords and access to my reddit account, So I will not be sharing the data unedited. 

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='plLFnSdBy7b8ZQ', client_secret='_fv-EVVpz_m4iekd9a2EFsfJ66E',
                     username=base64.b64decode('UHJpdmF0ZUFza1NjaWVuY2VCb3Q='), 
                     password=(base64.b64decode("SUxvdmVMaW5kc2V5MTIz")))
#askscience = reddit.subreddit('AskScience')

askscience = reddit.subreddit('askscience')

subs = ['physics', 'bio', 'med', 'geo', 'chem', 'astro']

data = pd.read_csv(r'askscience_Data.csv')
data = data.iloc[:,1:]

history = pd.read_csv(r'history.csv')
history = history.iloc[:, 1:]

dat = np.empty([data.shape[0], 300])
tags = Sub_treater(data.tag, subs)
tags = [tag.replace('other', 'Other') for tag in tags]

for i in range(data.shape[0]):
    temp = nlp(data.iloc[i,1]).vector
    for j in range(300):
        dat[i, j] = temp[j]
        

def Predict_post(Title):
    Title = nlp(Title).vector
    newdat = Feed_reduction(dat, tags, Title, nodes = 50)
    clf = svm.SVC(kernel = 'linear')
    clf.fit(newdat[0], tags)
    #print(newdat[0].shape)
    #print(newdat[1].shape)
    pred = clf.predict(newdat[1])
    return(pred[0])



print("Goodmorning General. I am up and running, Sir!")
for post in askscience.stream.submissions(skip_existing = True):
    

    history = pd.read_csv(r'history.csv')
    history = history.iloc[:, 1:]

    j = data.shape[0]
    i = history.shape[0]
    pred = Predict_post(post.title)
    history.loc[i,'id'] = post.id
    history.loc[i, 'title'] = post.title
    history.loc[i, 'prediction'] = pred
    history.loc[i, 'actual'] = post.link_flair_css_class
    if pred == post.link_flair_css_class:
        history.loc[i, 'correct'] = 1
        tags.append(post.link_flair_css_class)
    elif pred == 'Other' and post.link_flair_css_class not in tags:
        history.loc[i, 'correct'] = 1
        tags.append('Other')
    else:
        history.loc[i, 'correct'] = 0
        if post.link_flair_css_class in tags:
            tags.append(post.link_flair_css_class)
        else:
            tags.append('Other')
    print("\n")
    data.loc[j,:] = [post.id, post.title, post.link_flair_css_class]
    data.to_csv("askscience_Data.csv")
    history.loc[i, 'time'] = datetime.datetime.now().date()
    history.to_csv('history.csv')
    dat = np.vstack([dat, nlp(post.title).vector])
    if history.loc[i, 'correct'] == 1:
            print("CORRECT!!!!!!!! New post: {} \n with tag: {} and prediction {} \n My accuracy is now: {} \n".format(
                    post.title, 
                    post.link_flair_css_class, 
                    pred, 
                    round(sum(history['correct'])/history.shape[0], 2)*100))                                      
    else:
            print("WRONG!!!!!!!! New post: {} \n with tag: {} and prediction {} \n My accuracy is now: {} \n".format(
                    post.title, 
                    post.link_flair_css_class, 
                    pred, 
                    round(sum(history['correct'])/history.shape[0], 2)*100)) 

    
    

















