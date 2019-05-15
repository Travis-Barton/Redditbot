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
import itertools
import pandas as pd
from sklearn.metrics import confusion_matrix
from random import choice, sample
import warnings
from datetime import datetime
from datetime import date
from datetime import timedelta
import time
import random
from Reddit_instance import reddit

travsbots = reddit.subreddit('travsbots')
history = pd.read_csv(r'history.csv')

history = history.iloc[:,1:]
lastposts = 650

k= 0
while True:
    try:
        times  = datetime.now().weekday()
        if times == 0:
            history = pd.read_csv(r'history.csv')
            history = history.iloc[:,1:]
            print("I am about to post on {}".format(date.today()))
            row = random.sample(list(np.where(history['correct'] == 0)[0]), 8)
            numerofposts = (history.shape[0])
            reddit.subreddit('travsbots').submit('Private Askscience Bot for myweekly update, Sir!', selftext = 
                          '''I have been working hard to record the content of \
                          r/askscience and attempt to predict the tags of each post. \
                          My goal is to someday be a moderator there! They are not \
                          currently allowing robots to be moderators, but I will be \
                          attempting to prove to them that I am reliable! I try to \
                          predict the main 6 catagories, and leave the less populus \
                          ones to be classified as \'other\' but that might change \
                          as I gather more data! \n This week, I have classified {} posts, and am doing more \
                          everyday. In general, I have an accuracy around {}% and \
                          have classified {} posts in total. I use Google's [natural language proccessing \
                          auto ML software](www.console.cloud.google.com/natural-language) to classify posts.\
                          Travis (my creator) has developed another method for dealing with this data that involves a natural \
                          language proccessing technique called ['word vectors'](www.nlp.stanford.edu/projects/glove/), \
                          a variable reduction method \
                          invented by Travis Barton called Feed Networks, and SVM. You can read all
                          about it under Passion Projects on his 
                          website: [www.wbbpredictions.com](http://www.wbbpredictions.com) \n Some examples of trouble posts are: \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__ \
                          \n \n \"{}\" classified as __{}__ when the mods classified it as __{}__
                          '''.format(
                          history.shape[0]-lastposts, 
                          np.round(sum(history['correct'])/history.shape[0]*100, 2), 
                          history.shape[0], 
                          history.iloc[row[0], 1], 
                          history.loc[row[0], 'prediction'], 
                          history.loc[row[0], 'actual'], 
                          history.iloc[row[1], 1], 
                          history.loc[row[1], 'prediction'],
                          history.loc[row[1], 'actual'],
                          history.iloc[row[2], 1], 
                          history.loc[row[2], 'prediction'],
                          history.loc[row[2], 'actual'],
                          history.iloc[row[3], 1], 
                          history.loc[row[3], 'prediction'],
                          history.loc[row[3], 'actual'],
                          history.iloc[row[4], 1], 
                          history.loc[row[4], 'prediction'],
                          history.loc[row[4], 'actual'],
                          history.iloc[row[5], 1], 
                          history.loc[row[5], 'prediction'],
                          history.loc[row[5], 'actual'],
                          history.iloc[row[6], 1], 
                          history.loc[row[6], 'prediction'],
                          history.loc[row[6], 'actual'],
                          history.iloc[row[7], 1], 
                          history.loc[row[7], 'prediction'],
                          history.loc[row[7], 'actual']
                          ))
            lastposts = history.shape[0]
            time.sleep(86400)
        else:
            time.sleep(86400)
    except Exception as e:
            print("I came accross an error Sir. I'll try restarting in 60 seconds: \n {} \n".format(e))
    time.sleep(60)


 






















