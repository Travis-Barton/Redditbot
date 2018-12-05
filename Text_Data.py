#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:10:03 2018

@author: travisbarton
"""

'''Getting the text data we need from reddit'''



import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from collections import Counter
import datetime
import time
import requests

#Setup... This contains passwords and access to my reddit account, So I will not be sharing the data unedited. 

reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username=base64.b64decode('bWF0aF9pc19teV9yZWxpZ2lvbg=='), 
                     password=(base64.b64decode("U2lyemlwcHkx")))
askscience = reddit.subreddit('AskScience')


def Update_Data(awww, Full_Data, lim):
     i = Full_Data.shape[0]
     test = awww.top('week', limit = lim)
     for post in test:
         if post.id not in list(Full_Data.iloc[:,0]):
             title = post.title
             tag = post.link_flair_css_class
             if 'best ' in str(tag):
                 tag = tag.replace('best ', '')
             elif 'lockdown ' in str(tag):
                 tag = tag.replace('lockdown ', '')
             Full_Data.loc[i,:] = [post.id, title.encode('ascii', 'ignore'), tag]
             i += 1
     return(Full_Data)

def Remove_Best(titles):
    i = 0
    for title in titles:
        if 'best ' in str(title):
            #print(title)
            titles[i] = title.replace('best ', '')
        elif 'lockdown ' in str(title):
            titles[i] = title.replace('lockdown ', '')
        i += 1
    return(titles)
def submissions_pushshift_praw(subreddit, start=None, end=None, limit=1000, extra_query=""):
    """
    A simple function that returns a list of PRAW submission objects during a particular period from a defined sub.
    This function serves as a replacement for the now deprecated PRAW `submissions()` method.

    :param subreddit: A subreddit name to fetch submissions from.
    :param start: A Unix time integer. Posts fetched will be AFTER this time. (default: None)
    :param end: A Unix time integer. Posts fetched will be BEFORE this time. (default: None)
    :param limit: There needs to be a defined limit of results (default: 100), or Pushshift will return only 25.
    :param extra_query: A query string is optional. If an extra_query string is not supplied, 
                        the function will just grab everything from the defined time period. (default: empty string)

    Submissions are yielded newest first.

    For more information on PRAW, see: https://github.com/praw-dev/praw 
    For more information on Pushshift, see: https://github.com/pushshift/api
    """
    matching_praw_submissions = []

    # Default time values if none are defined (credit to u/bboe's PRAW `submissions()` for this section)
    utc_offset = 28800
    now = int(time.time())
    start = max(int(start) + utc_offset if start else 0, 0)
    end = min(int(end) if end else now, now) + utc_offset

    # Format our search link properly.
    search_link = ('https://api.pushshift.io/reddit/submission/search/'
                   '?subreddit={}&after={}&before={}&sort_type=score&sort=asc&limit={}&q={}')
    search_link = search_link.format(subreddit, start, end, limit, extra_query)

    # Get the data from Pushshift as JSON.
    retrieved_data = requests.get(search_link)
    returned_submissions = retrieved_data.json()['data']

    # Iterate over the returned submissions to convert them to PRAW submission objects.
    for submission in returned_submissions:

        # Take the ID, fetch the PRAW submission object, and append to our list
        praw_submission = reddit.submission(id=submission['id'])
        matching_praw_submissions.append(praw_submission)

    # Return all PRAW submissions that were obtained.
    return matching_praw_submissions



def Id_to_post(ids, reddit, data):
    dat = data
    #dat = pd.DataFrame(columns = ['id', 'Title', 'tag'])
    V = dat.shape[0]
    for idd in ids:
        if idd not in list(dat.iloc[:,0]):
            temp = reddit.submission(id=str(idd))
            title = temp.title
            print(title)
            tag = temp.link_flair_css_class
            if 'best ' in str(tag):
                tag = tag.replace('best ', '')
            elif 'lockdown ' in str(tag):
                tag = tag.replace('lockdown ', '')
            dat.loc[V, :] = ([idd, str(title.encode('ascii', 'ignore')), tag])
            V += 1
    return(dat)
            

## Dataset creation
data = pd.read_csv("/Users/travisbarton/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]
#data = pd.DataFrame(columns = ['id', 'Title', 'tag'])

# Filling the data
data = Update_Data(askscience, data, 10000)
#data.loc[:, 2] = Remove_Best(data.iloc[:, 2])
#print(Counter(data.iloc[:, 2]))
data.to_csv("askscience_Data.csv")

print(len(data.iloc[:, 0]) == len(set(data.iloc[:, 0])))
print(Counter(data.iloc[:, 2]))
