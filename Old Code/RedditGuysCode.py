#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 18:21:55 2018

@author: travisbarton
"""

import requests
import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from collections import Counter
import time


# Authentication: http://praw.readthedocs.io/en/latest/getting_started/authentication.html
from Reddit_instance import reddit

def submissions_pushshift_praw(subreddit, start=None, end=None, limit=100, extra_query=""):
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


test = submissions_pushshift_praw("askscience", start = time.time() - 126144000, limit = 10000)
data = pd.read_csv('~/Documents/GitHub/Redditbot/askscience_Data.csv')
data = data.iloc[:,1:]
k = 0
for post in test:
    temp = reddit.submission(id = post.id)
    if (data['id'].str.contains(post.id).any() == False and temp.link_flair_css_class != None):
        #print(temp.link_flair_css_class)
        i = data.shape[0]
        data.loc[i,:] = [temp.id, temp.title, temp.link_flair_css_class]
        k += 1
    if k == 1000:
        break
    

data.to_csv('temp.csv')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    