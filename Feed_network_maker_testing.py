# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:44:50 2018

@author: sivar
"""



#easydata = pd.read_csv("/Users/travisbarton/Documents/GitHub/Redditbot/VDifferentData.csv")
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



easydat = np.empty([easydata.shape[0],301])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
tags = easydata.tag 
easydat[:,300] =easydata.tag
dat = easydat.copy()


        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.15, 
                                                    random_state=RS)   


y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  

y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   

temp = Binary_network(X_train, y_train, X_test, "dont matter yet", .1, 50, 15, 30)


train_res = np.round(temp[0]).astype(int)
test_res = np.round(temp[1]).astype(int)


Pred_to_num(y_test)[0:10]

1-sum(train_res == Pred_to_num(y_train))/dat.shape[0]
1-sum(test_res == Pred_to_num(y_test))/dat.shape[0]
plot_confusion_matrix(confusion_matrix(Pred_to_num(y_train), train_res), [0,1], normalize = True, title = "Is test good?")
plot_confusion_matrix(confusion_matrix(Pred_to_num(y_test), test_res), [0,1], normalize = True, title = "Is test good?")


#When you pick up next time. You are working on integrating the binary networks
#you have it returning the probability of being in the first column rn.
# next time erase label paremeter and input x_test parameter



############### test #2 
easydat = np.empty([easydata.shape[0],300])    

for i in range(easydat.shape[0]):
    vecs = nlp(easydata.iloc[i,1]).vector
    for j in range(300):
        easydat[i,j] = vecs[j]
    
tags = easydata.tag 

tag = []
for i in range(len(tags)):
    if tags.iloc[i] == 1.0:
        tag.append('Aww')
    elif tags.iloc[i] == 2.0:
        tag.append('Politics')
    else:
        tag.append("ST")

dat = easydat.copy()

onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    tag, 
                                                    test_size=0.25, 
                                                    random_state=RS)   







results = []
results = Feed_reduction(X_train, y_train, X_test, np.unique(y_train), nodes = 50)
new_X = results[0]
new_X_test = results[1]


# Feed networks
clf = svm.SVC(gamma='scale')
clf.fit(new_X, y_train)  
drumroll = clf.predict(new_X_test)

print('the accuracy of Feed networks: {}'.format(sum(drumroll == y_test)/len(y_test)*100))



# SVM
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)  
drumroll2 = clf.predict(X_test)

print('the accuracy of just SVM is: {}'.format(sum(drumroll2 == y_test)/len(y_test)*100))


# Full network

y = pd.factorize(y_train)[0]
y = y.reshape(len(y), 1).astype(int)        
y = onehot_encoder.fit_transform(y)

model = Sequential()

model.add(Dense(50, input_dim = X_train.shape[1], activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha = .001))
model.add(Dense(3, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
    #filepath="Best_{}.hdf5".format(label)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
     #                            save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
model_history = model.fit(X_train, y, 
                          epochs=20, batch_size=30, 
                          verbose = 0, validation_split = .2)
Full_results = Pred_to_num(model.predict(X_test))
for i in range(len(Full_results)):
    if Full_results[i] == 1:
        Full_results[i] = 2
    elif Full_results[i] == 2:
        Full_results[i] == 1
    else:
        pass


sum(Full_results == pd.factorize(y_test)[0])/len(y_test)


yt = pd.factorize(y_test)[0]
yt = yt.reshape(len(yt), 1).astype(int)        
yt = onehot_encoder.fit_transform(yt)

model.evaluate(X_test, yt)



