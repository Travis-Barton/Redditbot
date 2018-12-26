# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 15:58:46 2018

@author: sivar
"""


# This is the testing happening with a formalized dataset. I am very exited!


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dat1 = unpickle('cifar-10-batches-py/data_batch_1')
dat2 = unpickle('cifar-10-batches-py/data_batch_2')
dat3 = unpickle('cifar-10-batches-py/data_batch_3')
dat4 = unpickle('cifar-10-batches-py/data_batch_4')
dat5 = unpickle('cifar-10-batches-py/data_batch_5')
dat = np.vstack([dat1[b'data'], dat2[b'data'], dat3[b'data'], dat4[b'data'], dat5[b'data']])/256
tags = np.hstack([dat1[b'labels'], dat2[b'labels'], dat3[b'labels'], dat4[b'labels'], dat5[b'labels']])
tags.shape
dat.shape



test = unpickle('cifar-10-batches-py/test_batch')
test_dat = test[b'data']/256
test_tags = np.array(test[b'labels'])


new_dat = Feed_reduction(x, y, x_test, labels = np.unique(y), val_split = .3, nodes = 1000, epochs = 100,batch_size = 5000, verbose = 2)
mod = svm.SVC(kernel = 'linear')
mod.fit(new_dat[0], tags)
preds = mod.predict(new_dat[1])
np.unique(preds)





##### Testing and debugging



def Binary_network(X, Y, X_test, label, val_split, nodes, epochs, batch_size, verbose = 0, sparse = False):
    
    model = Sequential()

    model.add(Dense(nodes, input_dim = X.shape[1], activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(.5))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(.4))
    model.add(Dense(nodes, activation = 'linear'))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dense(2, activation = 'softmax'))        
            
    model.compile(loss='binary_crossentropy', 
                  optimizer='sgd', 
                  metrics=['accuracy'])
    #filepath="Best_{}.hdf5".format(label)
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
     #                            save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    model_history = model.fit(X, Y, 
                              epochs=epochs, batch_size=batch_size, 
                              verbose = verbose, validation_split = val_split)
    physpreds = model.predict(X)
    confm = confusion_matrix(Pred_to_num(Y), Pred_to_num(physpreds))
    plot_confusion_matrix(confm, [0,1], normalize = True, title = "?")
    
    print(X_test)
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

    if (X_test.ndim == 1):
        X_test = np.array([X_test])
    return([model.predict(X)[:,0], model.predict(X_test)[:,0]])




onehot_encoder = OneHotEncoder(sparse=False) 


x = dat
y = tags
x_test = test_dat
#if type(y[0]) != 'str'

index = np.where(y == 1)[0]
index2 = np.where(y!= 1)[0]
x2 = np.vstack([[dat[item,:] for item in index2], random.sample([dat[item,:] for item in index], 4000)])
x2 = np.vstack([dat, random.sample([dat[item,:] for item in index], 4000)])
y2 = np.append(y, np.repeat(1, 4000))
y


y2 = Sub_treater(y2, 1)
y2 = pd.factorize(y2)[0]
y2 = y2.reshape(len(y2), 1).astype(int)        
y2 = onehot_encoder.fit_transform(y2)
temp = Binary_network(x2, y2, x_test, 'whatever', val_split = .2, nodes = 800, epochs = 100, batch_size = 2000, verbose = 1)


confm = confusion_matrix(Pred_to_num(y), Pred_to_num(temp[0]))
plot_confusion_matrix(confm, [0,1], normalize = True, title = "?")
print(X_test)


'''
Where you left off:
    There is too much of a pull from the smaller class. The imbalanced data causes issues.
    Find a way to balance the data somehow.

    

'''

datt = datasets.load_digits()

x = datt['data']/16
y = datt.target
x_test = datt.images/16
y_test = datt.target_names

train = pd.read_csv("/Users/travisbarton/Documents/GitHub/Feed Network Testing/training.csv")
test = pd.read_csv("/Users/travisbarton/Documents/GitHub/Feed Network Testing/testing.csv")
y = train.iloc[:,784]
y_test = test.iloc[:,784]

train = train.iloc[:,:784]/256
test = test.iloc[:,:784]/256


new_dat = Feed_reduction(train, y, test, labels = np.unique(y), val_split = .3, nodes = 100, epochs = 20,batch_size = 300, verbose = 2)


svmtime = svm.SVC(gamma = .001)
svmtime.fit(new_dat[0], y)
predsvm = svmtime.predict(new_dat[1])
1 - sum(predsvm == y_test)/len(y_test)

new_train = np.array(new_dat[0])


svmm = svm.SVC(gamma = .001)
svmm.fit(train.values, y.values)
predssvm2 = svmm.predict(test.values)
1 - sum(predssvm2 == y_test)/len(y_test)

PCA
