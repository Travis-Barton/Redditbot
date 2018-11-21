### Data Pipeline

##Set ups
MAX_SEQUENCE_LENGTH <- 100
MAX_NUM_WORDS <- 10000
EMBEDDING_DIM <- 300
VALIDATION_SPLIT = .4
sentences <- read_csv("~/Redditbot/askscience_Data.csv")
sentences[which(is.na(sentences), arr.ind = T)] = 'meta'

embeddings_index <- read.table("~/Redditbot/glove.6B/glove.6B.300d.txt", sep = " ", header = FALSE,
                                 quote = NULL, comment.char = "", row.names = 1,
                                 nrows = -1)
texts <- sentences$Title # text samples
labels <- as.integer(factor(sentences$tag, labels = c(1:13))) # label ids
labels_index <- list()  # dictionary: label name to numeric id
temp = factor(sentences$tag)
for(i in 1:13)
{
  labels_index[[levels(temp)[i]]] = i
}

labels <- to_categorical(labels)[,-1]
tokenizer <- text_tokenizer(num_words=MAX_NUM_WORDS, lower = T, filters = c("i", "what", "who", "when", "why", "where"))
tokenizer %>% fit_text_tokenizer(texts)

#Sequences is important. 
#Its the Bag of words style word vectors... kinda. 
#its the most populus 1000 words in the dictionary
sequences <- texts_to_sequences(tokenizer, texts)

#Word index is the location of words in our dictionary
word_index <- tokenizer$word_index

#Data is our data standardized to a given length
data <- pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


#Lets shuffle the data, and then turn it into train and test sets
#shuffle
indices <- 1:nrow(data)
indices <- sample(indices)
data <- data[indices,]
labels <- labels[indices,]

#split
num_validation_samples <- as.integer(VALIDATION_SPLIT * nrow(data))
x_train <- data[-(1:num_validation_samples),]
y_train <- labels[-(1:num_validation_samples),]
x_val <- data[1:num_validation_samples,]
y_val <- labels[1:num_validation_samples,]

#next we need to make the weights that turn our data into the 
# 100d sentence vectors
num_words <- min(MAX_NUM_WORDS, length(word_index) + 1)
prepare_embedding_matrix <- function() {
  embedding_matrix <- matrix(0L, nrow = num_words, ncol = EMBEDDING_DIM)
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index >= MAX_NUM_WORDS)
      next
    embedding_vector <- as.numeric(embeddings_index[word,])
    if (!is.null(embedding_vector)) {
      # words not found in embedding index will be all-zeros.
      embedding_matrix[index,] <- embedding_vector
    }
  }
  embedding_matrix
}

#embeddings_matrix is the embedding of the top 20000 words
#in the dictionary
embedding_matrix <- prepare_embedding_matrix()

###NOW COMES THE NETWORK :) 
reset_states(model)
#First Layer that translates bag'o'words into embedding
embedding_layer <- layer_embedding(
  input_dim = num_words,
  output_dim = EMBEDDING_DIM,
  weights = list(embedding_matrix),
  input_length = MAX_SEQUENCE_LENGTH,
  trainable = TRUE
)

#Input layer
sequence_input <- layer_input(shape = list(MAX_SEQUENCE_LENGTH), dtype='int32')

#network structure
preds <- sequence_input %>%
  embedding_layer %>% 
  layer_conv_1d(filters = 64, kernel_size = 5, activation = 'sigmoid') %>% 
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'sigmoid') %>% 
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'sigmoid') %>% 
  layer_dense(units = 13, activation = 'softmax')
  #layer_dense(units = length(labels_index), activation = 'softmax')

preds  <- sequence_input %>%
  embedding_layer %>% 
  layer_flatten() %>% 
  layer_dropout(.2) %>%
  layer_dense(units = 32, activation = 'sigmoid') %>%
  layer_dense(units = 13, activation = 'softmax')
#Model Creation
model <- keras_model(sequence_input, preds)

#compiler
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')  
)

#here comes the fit!
summary(model)
history <- model %>% fit(
  x_train, y_train,
  batch_size = 500,
  epochs = 100,
  validation_data = list(x_val, y_val),
  verbose=1
)


history$params

predictions <- predict(model, x_val, 200, 1)
finalpreds <- {}
truepredds <- {}
for(i in 1:nrow(x_val))
{
  finalpreds[i] = which(predictions[i,] == max(predictions[i,]))
  truepredds[i] = which(y_val[i,] == 1)
}

percent(table(finalpreds, truepredds))
Percent(finalpreds, truepredds)
