#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 17:25:07 2018

@author: travisbarton
"""
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
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='bepL4c1oHsSEIA', client_secret='Mj6pnYQS1QB6Vh1Fyt1cXSb1EF8',
                     username=base64.b64decode('RHJpbGxTYXJnZW50Qm90'), 
                     password=(base64.b64decode("SUxvdmVMaW5kc2V5MTIz")))


mysub = reddit.subreddit('travsbots')


for post in mysub.stream.submissions(skip_existing = True):
    post.upvote()
    print(post.title)
    if 'asksciencebot' and 'report' in post.title:
        history = pd.read_csv(r'history.csv')
        history = history.iloc[:, 1:]
        correct = sum(history['correct'])
        numpreds = history.shape[0]
        if numpreds == 0:
            post.reply('Private Askscience Bot is still awaiting its first post, Sir!')
        else:
            post.reply('Private Askscience Bot has achieved an accuracy of %{} out of {} posts, Sir!'.format(
                    np.round(correct/numpreds, 4)*100, numpreds))
        

