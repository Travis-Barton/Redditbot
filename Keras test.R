library(keras)
use_condaenv("r-tensorflow")
library(dplyr)
library(onehot)
library(scales)

train[,1:256] = (train[,1:256]+1)/2
test[,1:256] = (test[,1:256]+1)/2

train[,1:256] = apply(train[,1:256], 2, rescale)
test[,1:256] = apply(test[,1:256], 2, rescale)



# input shape is the vocabulary count used for the movie reviews (10,000 words)
reset_states(model)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(256)) %>%
  layer_dropout(rate = .4) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = .4) %>%
  layer_dense(units = 10, activation= 'softmax')


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = .001),
  metrics = c('accuracy')
)

history <- model %>% fit(
  as.matrix(train[,1:256]),
  (y_labs),
  epochs = 30,
  batch_size = 64,
  validation_data = list(as.matrix(test[,1:256]), y_labs_test),
  verbose=1
)
score <- model %>% 
  evaluate(as.matrix(test[,1:256]), y_labs_test, batch_size=128)

score

