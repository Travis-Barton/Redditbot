# Feed network for R

library(keras)




Binary_Network <- function(X, Y, X_test, val_split, nodes, epochs, batch_size, verbose = 0) {
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = nodes, input_shape = ncol(X)) %>%
    layer_activation_leaky_relu(alpha = .001) %>%
    layer_dropout(.4) %>%
    layer_dense(units = nodes) %>%
    layer_activation_leaky_relu(.001) %>%
    layer_dense(units = 2, activation = "softmax")
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
  history <- model %>% fit(
    X, Y, 
    epochs = epochs,
    batch_size = batch_size,
    validation_split = val_split
  )
  
  train = model %>% predict_proba(X)
  test = model %>% predict_proba(X_test)
  dat = list(train = train, test = test)
  return(dat)
}

Feed_Reduction <- function(X, Y, X_test, val_split = .1, nodes = NULL, epochs = 15, batch_size = 30, verbose = 0) {
  if(is.null(nodes) == TRUE){
    nodes = round(ncol(X)/4)
  }
  labels = sort(unique(Y), decreasing = F)
  final_train = matrix(0, nrow = nrow(X), ncol = length(labels))
  final_test = matrix(0, nrow = nrow(X_test), ncol = length(labels))
  i = 1
  for(label in labels){
    index = which(Y == label)
    y <- Y
    y[index] = 1
    y[-index] = 0
    y = to_categorical(y, 2)
    temp = Binary_Network(X, y, X_test, val_split, nodes, epochs, batch_size, verbose)
    final_train[, i] = temp$train[,1]
    final_test[, i] = temp$test[,1]
    i = i + 1
    
  }
  
  return(list(train = final_train, test = final_test))
}





