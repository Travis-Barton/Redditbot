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
from Feed_network_maker import plot_confusion_matrix, Sub_treater, Binary_network, Feed_reduction
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



for i in range(data.shape[0]):
    temp = nlp(data.iloc[i,1]).vector
    for j in range(300):
        dat[i, j] = temp[j]
        

def Predict_post(Title):
    Title = nlp(Title).vector
    #Title = Title.reshape([300, 1])
    newdat = Feed_reduction(dat, tags, Title, nodes = 50)
    clf = svm.SVC(kernel = 'linear')
    clf.fit(newdat[0], tags)
    pred = clf.predict(newdat[1])
    return(pred)

tites = 'I love Physics'
HappyHappyJoyJoy = Predict_post(tites)


for post in askscience.stream.submissions(skip_existing = True):
    i = data.shape[0]
    pred = Predict_post(post)
    history['id'] = post.id
    history['prediction'] = pred
    if pred == post.tag:
        history['actual'] = pred
        history['correct'] = 1
    elif pred == 'Other' and post.tag in Other_tags:
        history['actual'] = post.tag
        history['correct'] = 1
    else:
        history['actual'] = post.tag
        history['correct'] = 0
    data.iloc[i,:] = [post.id, post.title, post.tag]
    data.to_csv("askscience_Data.csv")
    history.to_csv('history.csv')

















