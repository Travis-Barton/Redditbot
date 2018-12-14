#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:37:38 2018

@author: travisbarton
"""
subs = ['physics', 'bio', 'med', 'geo', 'chem', 'astro']
# Comparisons
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]
Layer2_Spacy_vector = Turn_into_Spacy(data)
data.tag = Sub_treater(data.tag, subs)


dat = np.empty([(data.shape[0]), 301])

for i in range(data.shape[0]):
    for j in range(300):
        dat[i,j] = Layer2_Spacy_vector[i][j]
        
dat[:, 300] = pd.factorize(data.tag)[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[:,:300], 
                                                    dat[:,300], 
                                                    test_size=0.25, 
                                                    random_state=RS)   


### SVM
from sklearn import svm
clf = svm.SVC(gamma = 'scale', kernel = 'linear')
clf.fit(X_train, y_train)
svmpreds = clf.predict(X_test)
len(np.where(svmpreds == y_test)[0])/len(y_test)


confm = confusion_matrix(y_test, svmpreds)
confm
plot_confusion_matrix(confm, np.hstack([subs, 'other']), normalize = True, title = "Which subreddit?")

#### Argmax

argpreds = []

for i in range(Final_x_test.shape[0]):
    row = Final_x_test[i,:]
    row = (max(np.where(row == max(row))))
    row = min(row)
    argpreds.append(row)

len(np.where(argpreds == y_test)[0])/len(y_test)


#### Argmax SVM

clf = svm.SVC(gamma = 'scale', kernel = 'linear', degree = 3)
clf.fit(Final_x_train, y_train)
svmpreds = clf.predict(Final_x_test)
len(np.where(svmpreds == y_test)[0])/len(y_test)


### Full network

y_train = y_train.reshape(len(y_train), 1).astype(int)
y_test = y_test.reshape(len(y_test), 1).astype(int)  


y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)   



model = Sequential()

model.add(Dense(50, input_dim = 300, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.4))
model.add(Dense(100, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(.5))
model.add(Dense(50, activation = 'linear'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(7, activation = 'softmax'))        
        
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

filepath="Full_Model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]




model_history = model.fit(X_train, y_train, epochs=30, batch_size=30, 
                          verbose = 1,
                          validation_data =[X_test, y_test],
                          callbacks = callbacks_list)

plt.figure()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title("Accuracy")
plt.xticks(range(20))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(212)
plt.title("\nLoss")
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.xticks(range(20))


model.load_weights("Full_Model.hdf5")

comppreds = model.predict(X_test)


Percent(y_test, Pred_to_num(comppreds))
confm = confusion_matrix(Pred_to_num(y_test), Pred_to_num(comppreds))
confm
plot_confusion_matrix(confm, np.hstack([subs, 'other']), normalize = True, title = "Which subreddit?")

