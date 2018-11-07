#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:16:25 2018

@author: travisbarton
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import spacy
import numpy as np
Layer1Data = pd.read_csv("/Users/travisbarton/Redditbot/Training_data.csv")
Layer1Data = Layer1Data.iloc[:, 1:]


Layer2Data = pd.read_csv("/Users/travisbarton/Redditbot/askscience_Data.csv")
Layer2Data = Layer2Data.iloc[:, 1:]

################################### Precare ##########################################

for i in range(Layer2Data.shape[0]):
    point = Layer2Data.iloc[i, 1]
    if point[0:2] == "b'" or point[0:2] == 'b"':
        point = point[2:]
        point.replace("\'", "")
        point.replace('\"', '')
    Layer2Data.iloc[i, 1] = point


for i in range(Layer1Data.shape[0]):
    point = Layer1Data.iloc[i, 1]
    if point[0:2] == "b'" or point[0:2] == 'b"':
        point = point[2:]
        point.replace("\'", "")
        point.replace('\"', '')
    Layer1Data.iloc[i, 1] = point



################################### Bag of Words ##########################################

def Turn_into_BOW(data, tags = False):
    docs = []
    for i in range(data.shape[0]):
        temp  = str(data.iloc[i, 1])
        print(i)
        if("'" in temp):
            temp.replace("'", '')
        if((temp[2] == '[' or temp[0] == '[')):
            for i in range(len(temp)):
                if(temp[i] == ']'):
                    temp = temp[i+1:]
                    break
        docs.append(temp)
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(docs)
    #print(vectorizer.vocabulary_)

    textvectors = [[] for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        textvectors[i] = vectorizer.transform([docs[i]]).toarray()
    return(textvectors)
        

Layer1_BOW_vector = Turn_into_BOW(Layer1Data, tags = True)
Layer2_BOW_vector = Turn_into_BOW(Layer2Data)



################################### Spacy ##########################################


nlp = spacy.load('en_vectors_web_lg')

def Turn_into_Spacy(data, tags = False):
    docs = []
    for i in range(data.shape[0]):
        temp  = (data.iloc[i, 1])
        if(str("'") in str(temp)):
            temp.replace(str("'"), "")
        if((temp[2] == '[' or temp[0] == '[')):
            for i in range(len(temp)):
                if(temp[i] == ']'):
                    temp = str(temp[i+1:])
                    break
        temp = temp[0:len(temp)-1]        
        docs.append(temp)
    textvectors = [[] for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        print(docs[i])
        textvectors[i] = nlp(docs[i]).vector
       
    return(textvectors)


Layer2_Spacy_vector = Turn_into_Spacy(Layer2Data)
Layer1_Spacy_vector = Turn_into_Spacy(Layer1Data)
np.savetxt('Layer1.txt',Layer1_Spacy_vector, delimiter=', ', fmt='%12.8f')

Layer1Data.iloc[:, 2].to_csv('labels.csv')
#np.savetxt('test.txt',data, delimiter=', ', fmt='%12.8f')





















