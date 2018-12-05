#### Basic Network ####
library(keras)
reset_states(model)
model4 <- keras_model_sequential()

model4 %>%
  layer_dense(units = 50, activation = 'sigmoid', input_shape = c(50)) %>%
  layer_dense(units= 20, activation = 'sigmoid') %>%
  layer_dense(units = 12, activation = 'softmax')

model4 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')  
)

summary(model4)
history <- model4 %>% fit(
  as.matrix(train_glove), y_train_glove,
  batch_size = 10,
  epochs = 100,
  validation_data = list(as.matrix(test_glove), y_test_glove),
  verbose=1
)
