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
import pandas as pd
import warnings
from Reddit_instance import reddit


mysub = reddit.subreddit('travsbots')

while True:
    try:
        for post in mysub.stream.submissions(skip_existing = True):
            post.upvote()
            print(post.title)
            if ('asksciencebot' or 'askscience bot') and 'report' in post.title:
                history = pd.read_csv(r'history.csv')
                history = history.iloc[:, 1:]
                correct = sum(history['correct'])
                numpreds = history.shape[0]
                if numpreds == 0:
                    post.reply('Private Askscience Bot is still awaiting its first post, Sir!')
                else:
                    post.reply('Private Askscience Bot has achieved an accuracy of {}% out of {} posts, Sir! \n In his last 100 posts, he has achieved an accuracy of {}%.'.format(
                            np.round(correct/numpreds, 4)*100, numpreds,
                            round(sum(history.loc[(history.shape[0]-101):(history.shape[0]-1), 'correct'])/100, 4)*100))
    except Exception as e:
        print("I came accross an error Sir. I'll try restarting in 60 seconds: \n {} \n".format(e))
    time.sleep(60)

