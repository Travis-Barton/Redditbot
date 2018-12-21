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


































