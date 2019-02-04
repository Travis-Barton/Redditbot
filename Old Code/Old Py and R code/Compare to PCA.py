# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:51:28 2018

@author: sivar
"""


''' next try comparing it agiasnt PCA. Try the three following things:
    1) Just normal
    2) Feed network
    3) PCA
    4)PCA + Feed network
    5) Feed network + PCA
'''
from sklearn import svm

# Data for everyone

easydata = pd.read_csv(r"C:\Users\sivar\OneDrive\Documents\GitHub\Redditbot\VDifferentData.csv")
easydata.columns = ['id', 'title', 'tag']
reddit = praw.Reddit(user_agent='Comment Extraction (by /u/USERNAME)',
                     client_id='b8unlbKK1rWOow', client_secret='FuFwla268qevA5Ju1MgRPs2Sihg',
                     username=base64.b64decode('bWF0aF9pc19teV9yZWxpZ2lvbg=='), 
                     password=(base64.b64decode("U2lyemlwcHkx")))
ST = reddit.subreddit('showerthoughts')
i = easydata.shape[0]
for post in ST.top("all", limit = 1000):
    easydata.loc[i,:] = [i, post.title, 3]
    i = i+1




easydat = np.empty([easydata.shape[0],300])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
tags = easydata.tag 



X_train, X_test, y_train, y_test = train_test_split(easydat[:,:300], 
                                                    np.array(tags), 
                                                    test_size=0.25, 
                                                    random_state=RS)   








# 1)   

clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

sum(preds == y_test)/X_test.shape[0]



# 2) 

onehot_encoder = OneHotEncoder(sparse=False)       

#y = pd.factorize(y_train)[0]
y = y_train.reshape(len(y_train), 1).astype(int)        
y = onehot_encoder.fit_transform(y)





newdata = Feed_reduction(X_train, y_train, X_test, labels = np.unique(y_train), nodes = 50)

clf2 = svm.SVC(kernel = 'linear')
clf2.fit(newdata[0], y_train)

preds2 = clf2.predict(newdata[1])
sum(preds2 == y_test)/len(y_test)


# 3) 

import numpy as np
from sklearn.decomposition import PCA


PCAmod = PCA(n_components=.95)
PCAmod.fit(X_train)
PCAtrain = PCAmod.transform(X_train)
PCAtest = PCAmod.transform(X_test)


clf3 = svm.SVC(kernel = 'linear')
clf3.fit(PCAtrain, y_train)
preds3 = clf3.predict(PCAtest)
sum(preds3 == y_test)/len(y_test)


# 4) 

newdata2 = Feed_reduction(PCAtrain, y_train, PCAtest, labels = np.unique(y_train), nodes = 50)


clf4 = svm.SVC(kernel = 'linear')

clf4.fit(newdata2[0], y_train)
preds4 = clf4.predict(newdata2[1])
sum(preds4 == y_test)/len(y_test)






# 5)


newdata3 = Feed_reduction(X_train, y_train, X_test, labels = np.unique(y_train), nodes = 50)



PCAmod2 = PCA(n_components=.95)
PCAmod2.fit(newdata3[0])
PCAtrain = PCAmod2.transform(newdata3[0])
PCAtest = PCAmod2.transform(newdata3[1])



clf5 = svm.SVC(kernel = 'linear')

clf5.fit(PCAtrain, y_train)
preds4 = clf5.predict(PCAtest)
sum(preds4 == y_test)/len(y_test)




















