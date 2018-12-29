#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:46:25 2018

@author: travisbarton
"""

'''
this is the textual report structures for the various bots. Cut/paste and customize
according to the function of the bot
'''
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
from datetime import datetime
from datetime import timedelta
import time
import random
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='plLFnSdBy7b8ZQ', client_secret='_fv-EVVpz_m4iekd9a2EFsfJ66E',
                     username=base64.b64decode('UHJpdmF0ZUFza1NjaWVuY2VCb3Q='), 
                     password=(base64.b64decode("SUxvdmVMaW5kc2V5MTIz")))

travsbots = reddit.subreddit('travsbots')
history = pd.read_csv(r'history.csv')

history = history.iloc[:,1:]
lastposts = history.shape[0]

k= 0
while True:
    times  = datetime.now().weekday()
    if times == 0:
        history = pd.read_csv(r'history.csv')
        history = history.iloc[:,1:]
        row = random.sample(list(np.where(history.iloc[:,4] == 0)[0]), 2)
        numerofposts = (history.shape[0])
        reddit.subreddit('travsbots').submit('Private Askscience Bot for my
                        weekly update, Sir!', selftext = 
                      '''I have been working hard to record the content of 
                      r/askscience and attempt to predict the tags of each post. 
                      My goal is to someday be a moderator there! They are not 
                      currently allowing robots to be moderators, but I will be
                      attempting to prove to them that I am reliable! I try to 
                      predict the main 6 catagories, and leave the less populus
                      ones to be classified as \'other\' but that might change 
                      as I gather more data! 
                      \n \n
                      This week, I have classified {} posts, and am doing more 
                      everyday. In general, I have an accuracy around %{} and 
                      have classified {} posts in total. I use a natural 
                      language proccessing technique, a variable reduction method
                      invented by Travis Barton called Feed Networks and SVM in 
                      order to decide where each post belongs. You can read all
                      about it under Passion Projects on his 
                      website: [www.wbbpredictions.com](http://www.wbbpredictions.com)
                      \n \n \n 
                      Some examples of trouble posts are: 
                          \n \n 
                      \"{}\" classified as {} when the mods classified it as {}
                      \n \n 
                      and 
                      \n \n 
                      \"{}\" classified as {} when the mods classified it as {}
                      '''.format(
                      history.shape[0]-lastposts, 
                      np.round(sum(history['correct'])/history.shape[0]*100, 2), 
                      history.shape[0], 
                      history.iloc[row[0], 1], 
                      history.loc[row[0], 'prediction'], 
                      history.loc[row[0], 'actual'], 
                      history.iloc[row[1], 1], 
                      history.loc[row[1], 'prediction'],
                      history.loc[row[1], 'actual']
                      ))
        lastposts = history.shape[0]
        time.sleep(86400)
    else:
        time.sleep(86400)


 






















