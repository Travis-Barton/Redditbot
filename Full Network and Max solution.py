#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:38:34 2018

@author: travisbarton
"""

############## Max Solution



from keras import initializers



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



full_data = np.empty([data.shape[0] - len(astropreds), 2])


#physics
model.load_weights("Physics_Models/weights-improvement-0.87.hdf5")

phys = model.predict(X_train)[:,0]
physpreds = model.predict(X_test[:,:300])[:,0]


#Bio
model.load_weights("Bio_Models/weights-improvement-0.90.hdf5")
bio = model.predict(X_train)[:,0]
biopreds = model.predict(X_test[:,:300])[:,0]

#Med
model.load_weights("Med_Models/weights-improvement-0.91.hdf5")
med = model.predict(X_train)[:,0]
medpreds = model.predict(X_test[:,:300])[:,0]

#Geo
model.load_weights("Geo_Models/weights-improvement-0.92.hdf5")
geo = model.predict(X_train)[:,0]
geopreds = model.predict(X_test[:,:300])[:,0]

#Chem
model.load_weights("Chem_Models/weights-improvement-0.94.hdf5")
chem = model.predict(X_train)[:,0]
chempreds = model.predict(X_test[:,:300])[:,0]

#Astro

model.load_weights("Astro_Models/weights-improvement-0.95.hdf5")
astro = model.predict(X_train)[:,0]
astropreds = model.predict(X_test[:,:300])[:,0]

#Other

model.load_weights("Other_Models/weights-improvement-0.89.hdf5")
other = model.predict(X_train)[:,0]
otherpreds = model.predict(X_test[:,:300])[:,0]


subs = ['physics', 'bio', 'med', 'geo', 'chem', 'astro']
data = pd.read_csv("/Users/travisbarton/Documents/Github/Redditbot/askscience_Data.csv")
data = data.iloc[:, 1:]
data.tag = Sub_treater(data.tag, subs)
dat[0:data.shape[0], 300] = pd.factorize(data.tag)[0]
dat[data.shape[0]:,300] = pd.factorize(noise.iloc[:,2])[0]
onehot_encoder = OneHotEncoder(sparse=False)       
        

X_train, X_test, y_train, y_test = train_test_split(dat[0:data.shape[0],:300], 
                                                    dat[0:data.shape[0],300], 
                                                    test_size=0.25, 
                                                    random_state=100)   


Final_x_train = np.vstack([phys, bio, med, geo, chem, astro, other]).T
Final_y_train = y_train

Final_x_test = np.vstack([physpreds, biopreds, medpreds, geopreds, 
                          chempreds, astropreds, otherpreds]).T
Final_y_test = y_test


        
Final_y_train = Final_y_train.reshape(len(Final_y_train), 1).astype(int)
Final_y_test = Final_y_test.reshape(len(Final_y_test), 1).astype(int)  

Final_y_train = onehot_encoder.fit_transform(Final_y_train)
Final_y_test = onehot_encoder.fit_transform(Final_y_test)   



model = Sequential()
model.add(Dense(12, input_dim = 7, activation = 'sigmoid'))
model.add(Dense(13, activation = 'sigmoid'))
model.add(Dropout(.4))
model.add(Dense(15, activation = 'sigmoid'))
model.add(Dropout(.5))
model.add(Dense(7, activation = 'softmax'))


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

filepath="Final_Model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model_history = model.fit(Final_x_train, Final_y_train, epochs=1000, batch_size=100, 
                          verbose = 1,
                          validation_data =[Final_x_test, Final_y_test], 
                          callbacks = callbacks_list)

plt.figure()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(211)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title("Accuracy")
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2)
plt.subplot(212)
plt.title("\nLoss")
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])

plt.savefig("Final Performance.png")



model.load_weights("Final_Model.hdf5")

Fullpreds = model.predict(Final_x_test)


Percent(Final_y_test, Fullpreds)        
confm = confusion_matrix(Pred_to_num(Final_y_test), Pred_to_num(Fullpreds))
confm
plot_confusion_matrix(confm, np.hstack([subs, 'other']), normalize = True, title = "Which Topic?")
