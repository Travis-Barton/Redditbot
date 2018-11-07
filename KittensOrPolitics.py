#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:49:58 2018

@author: travisbarton
"""
import pandas as pd
import numpy as np
import praw
import base64
import spacy
nlp = spacy.load('en_vectors_web_lg')

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username=base64.b64decode('bWF0aF9pc19teV9yZWxpZ2lvbg=='), 
                     password=(base64.b64decode("U2lyemlwcHkx")))
aww = reddit.subreddit('aww')
politics = reddit.subreddit('Politics')


temp = pd.DataFrame(columns = ["title", "sub"])
awwtop = aww.hot(limit = 1000)
politop = politics.hot(limit = 1000)
i = 0
for post in awwtop:
    temp.loc[i,"title"] = post.title
    temp.loc[i, "sub"] = 1
    i += 1
for post in politop:
    temp.loc[i,"title"] = post.title
    temp.loc[i, "sub"] = 2
    i += 1

word = np.empty([temp.shape[0], 301])
sentences = [[] for i in range(temp.shape[0])]
for row in range(temp.shape[0]):
    word[row,0] = str(temp.iloc[row, 1]) # add label
    sentences[row] = str(temp.iloc[row, 0]) # add sentence
    word[row,1:] = (nlp(temp.iloc[row, 0]).vector)
    

temp.to_csv("VDifferentData.csv")
np.savetxt('VDiff_vec.csv', word, fmt='%f', delimiter=',')
np.savetxt('Vdiff_sentences.csv',sentences, fmt= '%s', delimiter = ',')
doc = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")

#for token in doc:
 #   print(token, token.lemma, token.lemma_)