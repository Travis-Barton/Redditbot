#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:07:50 2018

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
import matplotlib

#Setup... This contains passwords and access to my reddit account, So I will not be sharing the data unedited. 

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username=base64.b64decode('bWF0aF9pc19teV9yZWxpZ2lvbg=='), 
                     password=(base64.b64decode("U2lyemlwcHkx")))
Phys = reddit.subreddit('Physics')
Bio = reddit.subreddit('Biology')
Med1 = reddit.subreddit('Medicine')
Med2 = reddit.subreddit('medical_news')
Geo1 = reddit.subreddit('Geology')
Geo2 = reddit.subreddit('EarthScience')
Geo3 = reddit.subreddit('Geoscience')
Astro = reddit.subreddit('Astronomy')
Chem = reddit.subreddit('Chemistry')
Eng1 = reddit.subreddit('Engineering')
Eng2 = reddit.subreddit('ReverseEngineering')
Neuro1 = reddit.subreddit('Neuro')
Neuro2 = reddit.subreddit('Neuroscience')
Soc1 = reddit.subreddit('Socialscience')
Soc2 = reddit.subreddit('Politicalscience')
Soc3 = reddit.subreddit('Economics')
Soc4 = reddit.subreddit('Archaeology')
Soc5 = reddit.subreddit('Anthropology')
Maths1 = reddit.subreddit('Math')
Maths2 = reddit.subreddit('statistics')
Computing1 = reddit.subreddit('computerscience')
Computing2 = reddit.subreddit('computing')
Computing3 = reddit.subreddit('programming')
Computing4 = reddit.subreddit('datascience')
Computing5 = reddit.subreddit('artificial')
Psyc = reddit.subreddit('psychology')

sub_list = [Phys, Bio, Med1, Med2, Geo1, Geo2, Geo3, Astro, Chem, Eng1, Eng2,
            Neuro1, Neuro2, Soc1, Soc2, Soc3, Soc4, Soc5, Maths1, Maths2, Computing1, Computing2,
            Computing3, Computing4, Computing5, Psyc]

sub_names = ['physics', 'bio', 'med', 'med', 'geo', 'geo', 'geo', 'astro', 'chem', 
             'eng', 'eng', 'neuro', 'neuro', 'soc', 'soc', 'soc','soc', 'soc', 
             'maths', 'maths', 'computing', 'computing', 'computing', 'computing', 
             'computing', 'psych']


def training_updater(sub, Full_Data, sub_name,lim = 10000):
     i = Full_Data.shape[0]
     test = sub.top('year', limit = lim)
     for post in test:
         if post.id not in list(Full_Data.iloc[:,0]):
             title = post.title
             temp = str(title.encode('ascii', 'ignore'))
             tag = sub_name
             Full_Data.loc[i,:] = [post.id, temp, tag]
             i += 1
     return(Full_Data)
    

def subreddit_looper(sub_lists, sub_names, data):
    j = 0
    for sub in sub_lists:
        print("{:>20} displayed as {:>10}".format(sub.display_name, sub_names[j]))
        training_updater(sub, data, sub_names[j])
        j += 1
        







#Training_data = pd.DataFrame(columns = ["id", "title", "sub"])
Training_data = pd.read_csv("/Users/travisbarton/Redditbot/Training_data.csv")
Training_data = Training_data.iloc[:, 1:]

#For Everyone
#subreddit_looper(sub_list, sub_names, Training_data)

#For Physics, bio and Med
subreddit_looper(sub_list[0:8], sub_names[0:8], Training_data)
#Training_data.iloc[7473,1] = str("Okay")
#Training_data.iloc[7547,1] = str("Period")
#Training_data.iloc[7627,1] = str("Okay, Thumbs up, Clap")
#Training_data.iloc[25358, 1] = str('Question Mark')
#Training_data.iloc[30070, 1] = str('Question Mark')

for i in range(Training_data.shape[0]):
    text = Training_data.iloc[i,1]
    text.replace("b'", "")
    text.replace("\'","")
    Training_data.iloc[i,1] = text

Training_data.to_csv("Training_data.csv")


letter_counts = Counter(Training_data.iloc[:, 2])
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')



























