#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:28:08 2018

@author: travisbarton
"""

import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter


data = pd.read_csv("/Users/travisbarton/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]


for i in range(data.shape[0]):
    if 'best ' in str(data.iloc[i, 2]):
        data.iloc[i, 2] = str(data.iloc[i,2]).replace('best ', '')
    elif 'lockdown ' in str(data.iloc[i, 2]):
        data.iloc[i, 2] = str(data.iloc[i, 2]).replace('lockdown ', '')
        

nlp = spacy.load('en_vectors_web_lg')


badkeys = ['soc', 'computing', 'psych', str('nan'), 'meta']


for i in range(data.shape[0]):
    if any(key in str(data.iloc[i, 2]) for key in badkeys):
        data.iloc[i, 2] = 'other'
        
physindex = (data.iloc[:,2] == "physics")
bioindex = data.iloc[:, 2] == "bio"
medindex = data.iloc[:, 2] == "med"
geoindex = data.iloc[:, 2] == "geo"
astroindex = data.iloc[:, 2] == "astro"
chemindex = data.iloc[:, 2] == "chem"
engindex = data.iloc[:,2] == "eng"
otherindex = data.iloc[:,2] == "other"
neuroindex = data.iloc[:,2] == "neuro"
mathsindex = data.iloc[:,2] == "maths"



datt = pd.concat([data.loc[physindex], data.loc[bioindex], data.loc[medindex], 
                  data.loc[geoindex], data.loc[astroindex], data.loc[chemindex],
                  data.loc[engindex], data.loc[otherindex], data.loc[neuroindex], 
                  data.loc[mathsindex]])
mat = np.empty([data.shape[0], data.shape[0]])


dat = list()
for sent in (datt.iloc[:,1]):
    dat.append(nlp(sent))



mat = np.empty([data.shape[0], 384])
labels = list()
for i in range(data.shape[0]):
    mat[i,:] = dat[i].vector
    labels.append(datt.iloc[i, 2])


lab = pd.DataFrame(labels)

lab.to_csv("label_data.csv")
np.savetxt('vector_data.csv', mat, fmt='%f', delimiter=',')













