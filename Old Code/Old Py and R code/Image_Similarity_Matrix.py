#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 12:56:06 2018

@author: travisbarton
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 17:09:52 2018

@author: travisbarton
"""
import re, requests, os, glob, sys
import urllib.request 
import base64, datetime
import  praw, prawcore
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
def Update_Data(awww, Full_Data, lim):
     i = 0
     for post in awww.top('all', limit = lim):
         if post.id not in list(Full_Data.iloc[:,0]):
             Full_Data.loc[i,:] = [post.id, post.title, Comment_Count(post), Avg_Karma(post), Get_Date(post).year, Get_Date(post).month, Get_Date(post).day, Get_Date(post).hour, Get_Date(post).minute,Get_Date(post).second, 'repost','image similarity', post.score]
         i += 1
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
    
def Get_Date(post):
    time = post.created
    return datetime.datetime.fromtimestamp(time)

#def Image_Similarity(post):
    
#SETUP
Passphrase = base64.b64decode("U2lyemlwcHkx")
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username='math_is_my_religion', password=str(Passphrase))
awww = reddit.subreddit('awww')

'''
#DATASET CREATION:   
Full_Data = pd.DataFrame(columns=['id', 'Title', 'Number of comments', 'avg_user_karma','post_year', 'post_month', 'post_day', 'post_hour', 'post_min','post_second', 'repost','image similarity', 'karma'])
#Full_Data = pd.read_csv("Aww_Data.csv")
#Full_Data = Full_Data.iloc[:, 1:]



# UPDATE YOUR DATA
Full_Data = Update_Data(awww, Full_Data, 10)
Full_Data.to_csv("Aww_Data.csv")
print(Full_Data.iloc[0, 5])
'''

#### HERE IS AN IDEA
''' Lets make an kmeans cluster for the photos (good, bad, avg) in aww then another variable can be
distance to means

'''


'''
def Create_Image_Means(subreddit):
    for post in subreddit.top('all', limit = 2):
        print(post.url)
    soup = BeautifulSoup(post.url, "lxml")
    matches = soup.select('.album-view-image-link a')
    imgurUrlPattern = re.compile(r'(http://i.imgur.com/(.*))(\?.*)?')
    Download_Image(imgurUrlPattern, "Test")

def Download_Image(imageUrl, localFileName):
    response = requests.get(imageUrl)
    if response.status_code == 200:
        print('Downloading %s...' % (localFileName))
    with open(localFileName, 'wb') as fo:
        for chunk in response.iter_content(4096):
            fo.write(chunk)
Create_Image_Means(awww)

'''




for post in awww.top('all', limit = 2):
    url = (post.url)
    print(file_name)
    urllib.request.urlopen(url)
    file_name = url.split("/")
    if len(file_name) == 0:
        print()
        file_name = re.findall("/(.*?)", url)
    file_name = file_name[-1]
    if "." not in file_name:
        file_name += ".jpg"
    print(file_name)
r = requests.get(url)
with open(file_name,"wb") as f:
    f.write(r.content)


















