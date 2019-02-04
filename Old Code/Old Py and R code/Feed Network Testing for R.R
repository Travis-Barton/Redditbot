#Feed Network Testing
dat <- dataset_mnist()
X_train = array_reshape(dat$train$x/255, c(nrow(dat$train$x/255), 784))
#y_train = to_categorical(dat$train$y, 10)
y_train = dat$train$y
X_test = array_reshape(dat$test$x/255, c(nrow(dat$test$x/255), 784))
#y_test =to_categorical(dat$test$y, 10)
y_test = dat$test$y


index_train = which(dat$train$y == 6 | dat$train$y == 5) %>%
  sample(., length(.))
index_test = which(dat$test$y == 6 | dat$test$y == 5) %>%
  sample(., length(.))

temp = Binary_Network(X_train[index_train,], y_train[index_train,c(7, 6)], X_test[index_test,], .3, 350, 30, 50)


temp2 = Feed_Reduction(X_train, y_train, X_test, val_split = .3, nodes = 350, 30, 50, verbose = 1)
temp2$train
y_train

unique(y_train)


library(e1071)
names(temp2$test) = names(temp2$train)
newdat = rbind(temp2$train, temp2$test)
mod = svm(y_train~newdat[c(1:60000),], kernel = 'linear', cost = 1, type = 'C-classification')
mod = svm(y_train~temp2$train, kernel = 'linear', cost = 1, type = 'C-classification')
preds = predict(mod, as.data.frame(temp2$test))
preds = predict(mod, as.matrix(newdat[-c(1:60000),]))
preds == y_test



newdat = as.data.frame(cbind(rbind(temp2$train, temp2$test), c(y_train, y_test)))
mod = svm(V11~., data = newdat, type = 'C-classification', kernel = 'linear', subset = c(1:60000))
preds = predict(mod, newdat[-c(1:60000),-11])

sum(preds == y_test)/length(y_test)

# 97% accuracy 


newdat2 = as.data.frame(cbind(rbind(X_train, X_test), c(y_train, y_test)))
mod2 = svm(V785~., data = newdat2, subset = c(1:60000), type = 'C-classification', kernel = 'linear')
preds2 = predict(mod2, newdat2[-c(1:60000),-785])
sum(preds2 == y_test)/length(y_test)
# plain is 93%


# PCA 


PCA = prcomp(X_train)
which(cumsum(PCA$sdev^2)/sum(PCA$sdev^2) < .95)
PCA_Train = X_train %*% PCA$rotation[,1:154]
PCA_Test = X_test %*% PCA$rotation[,1:154]
newdat3 = as.data.frame(cbind(rbind(PCA_Train, PCA_Test), c(y_train, y_test)))

mod3 = svm(V155~., data = newdat3, subset = c(1:60000), type = 'C-classification', kernel = 'linear')
preds3 = predict(mod3, newdat3[-c(1:60000), -155])
sum(preds3 == y_test)/length(y_test)

# 94%

### The winner is Feed Network Reduction!!!!!!
### Lets combine them!



