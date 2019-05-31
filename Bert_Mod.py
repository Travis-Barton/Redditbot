#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:41:10 2019

@author: travisbarton
"""

## Reddit remade


from Reddit_instance import reddit
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "jupyter-platform-008d64ef4bd4.json"
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
import pandas as pd
import numpy as np
import datetime
import time
import boto3
from Reddit_instance import s3



 
project_id = 'jupyter-platform'
model_id = 'TCN612148418293027439'

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

def disagree(post_title, truth, guess, link, i):
    if i == 5:
        return(0)
    title = u'[asksciencebot] ' + post_title
    body = u'My guess: ' + guess + u' \n \n The Mods: ' + truth + u'\n \n What do you think? __upvote__ if I did okay, __downvote__ otherwise. \n Remember, when I say \'other\' I mean one of these less used catagories [meta, soc, computing, psych, maths, neuro] \n \n check the post here:' + link         
    try:
        reddit.subreddit('travsbots').submit(title, selftext = body)
    except Exception as e:
        print(u"I came accross an error general, I think I am doing this too much. I'll try restarting in 60 seconds: \n {} \n".format(e))
        i += 1
        time.sleep(60*i )
        disagree(post_title, truth, guess, link, i)


def main():
    askscience = reddit.subreddit('askscience')
    subs = {'0':'astro', 
            '1':'bio', 
            '2':'chem', 
            '3':'eng', 
            '4':'geo', 
            '5':'med', 
            '6':'other', 
            '7':'physics'}
    obj = s3.get_object(Bucket='redditbot-storage', Key='askscience_Data.csv')
    data = pd.read_csv(obj['Body'])
#    data = pd.read_csv('askscience_Data.csv')
    data = data.iloc[:, 1:]
    print(data.shape)
    obj = s3.get_object(Bucket='redditbot-storage', Key='history.csv')
    history = pd.read_csv(obj['Body'])
#    history = pd.read_csv('history.csv')
    history = history.iloc[:,1:]
    print(history.shape)
    T = 0
    while True:
        try:
            for post in askscience.stream.submissions(skip_existing = False):
                if (data['id'].str.contains(post.id).any() == False):
                    i = data.shape[0]
                    j = history.shape[0]
                    print('\n {} \n'.format(post.title))
                    if post.link_flair_css_class == None:
                        post.link_flair_css_class = 'meta'
                    data.loc[i,:] = [post.id, post.title, post.link_flair_css_class]
                    pred = get_prediction(post.title, project_id, model_id).payload[0].display_name
                    if subs[pred] == post.link_flair_css_class or ((post.link_flair_css_class not in subs.values()) and subs[pred] == 'other'): 
                        print(u'Correct! \n My running acc is: {} % \n My overall acc is: {} %'.format(
                                sum(history['correct'][-100:]),
                                np.round(100*sum(history['correct'])/history.shape[0], 2)))
                        temp = 1.0
                    else:
                        print(u'Wrong! \n My running acc is: {} % \n My overall acc is: {} % \n Your guess: {} \n The truth: {}'.format(
                                sum(history['correct'][-100:]),
                                np.round(100*sum(history['correct'])/history.shape[0], 2),
                                subs[pred],
                                post.link_flair_css_class))
                        temp = 0.0
                        disagree(post.title, post.link_flair_css_class, subs[pred], post.shortlink, 0)
                       
                            
                    history.loc[j, :] = [post.id, post.title, subs[pred],
                                 post.link_flair_css_class, temp, datetime.datetime.now().date(),
                                 post.selftext]
                    history.to_csv(u'history.csv')
                    data.to_csv(u'askscience_Data.csv')
                    if l % 4 == 0:
                        filename = u'askscience_Data.csv'
                        bucket_name = u'redditbot-storage'
                        s3.upload_file(filename, bucket_name, filename)
                        filename = u'history.csv'
                        s3.upload_file(filename, bucket_name, filename)
                        l += 1

                    
        except Exception as e:
            print(u"I came accross an error general. I'll try restarting in 60 seconds: \n {} \n".format(e))
        time.sleep(60)
                

main()






    
    
    
    
