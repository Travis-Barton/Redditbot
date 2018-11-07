#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:09:52 2018

@author: travisbarton
"""

import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np

def Update_Data(awww, Full_Data, lim):
     i = 0
     test = awww.top('all', limit = lim)
     for post in test:
         
         if post.id not in list(Full_Data.iloc[:,0]):
             Matching_Urls.loc[i, 'url'] =  post.url
             title = post.title
             Full_Data.loc[i,:] = [post.id, title.encode('ascii', 'ignore'), post.num_comments, Total_Karma(post), Get_Date(post).year, Get_Date(post).month, Get_Date(post).day, Get_Date(post).hour, Get_Date(post).minute,Get_Date(post).second,'repost','image similarity', post.score]
         i += 1
         print(10-i)
     return(Full_Data)

def Comment_Count(post):
    post.comments.replace_more(limit = None)
    comment_que = post.comments[:]
    i = 0
    while comment_que:
        comment = comment_que[0]
        comment_que.pop(0)
        comment_que.extend(comment.replies)
        i += 1
    return(i)
    
def Avg_Karma(post):
    karma_by_sub = {}
    submissions_by_sub={}
    author = reddit.redditor(str(post.author))
    #print(author)
    try:
        submissions_from_author = author.submissions.new(limit = None)
        for thing in submissions_from_author:
            if thing.id != post.id:
                sub = thing.subreddit.display_name
                karma_by_sub[sub] = karma_by_sub.get(sub, 0) + thing.score
                submissions_by_sub[sub] = submissions_by_sub.get(sub, 0) + 1
            else:
                pass
        return(karma_by_sub[post.subreddit.display_name]/float(submissions_by_sub[post.subreddit.display_name]))
        
    except: return(float('nan'))   

def Total_Karma(post):
    author = post.author
    return(author.link_karma)
    
def Get_Date(post):
    time = post.created
    return datetime.datetime.fromtimestamp(time)

#def Image_Similarity(post):
    
#SETUP
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username='math_is_my_religion', password=str(base64.b64decode("U2lyemlwcHkx")))
awww = reddit.subreddit('aww')


#DATASET CREATION:   
Full_Data = pd.DataFrame(columns=['id', 'title', 'Number of comments', 'avg_user_karma','post_year', 'post_month', 'post_day', 'post_hour', 'post_min','post_second','repost','image similarity', 'karma'])
Matching_Urls = pd.DataFrame(columns =  ['url', 'picture'])
#Full_Data = pd.read_csv("Aww_Data.csv")
#Full_Data = Full_Data.iloc[:, 1:]



# UPDATE YOUR DATA
Full_Data = Update_Data(awww, Full_Data, 10)
Matching_Urls.iloc[:,0].to_csv('Matching_Urls.csv')
Full_Data.to_csv("Aww_Data.csv")



#### HERE IS AN IDEA
''' Lets make an kmeans cluster for the photos (good, bad, avg) in aww then another variable can be
distance to means

'''
