#Data Preproccessing using GLOVE in R
library(text2vec)
library(LilRhino)
library(readr)
library(tokenizers)
library(keras)
askscience_Data <- read_csv("~/Redditbot/askscience_Data.csv")
layer1data <- read_csv("~/Redditbot/Training_data.csv")
Create_Word_Vectors<- function(sentences, id)
{
  it <- itoken(sentences, tolower, word_tokenizer, id = id, n_chunks = 10)
  vocab <- create_vocabulary(it)
  vocab <- prune_vocabulary(vocab, term_count_min = 5L)
  vectorizer <- vocab_vectorizer(vocab)
  tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
  glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
  word_vectors_main = glove$fit_transform(tcm, n_iter = 20)
  word_vectors_context = glove$components
  word_vectors <- word_vectors_main + t(word_vectors_context)
  return(word_vectors)
}
Sentence_Vectorizer <- function(sent, method = "avg", word_vectors) 
{
  #Averages the word vectors in a sentence and outputs the new vector 
  #Dot product and averages are available
  if(method == "avg")
  {
    temp <- tokenize_words(sent, lowercase = T, simplify = T, stopwords = stopwords("en"))
    sent_vec = rep(0, 50)
    valid_words = 0
    for(i in 1:length(temp))
    {
      if(temp[i] %in% rownames(word_vectors))
         {
            valid_words = valid_words + 1
            sent_vec = sent_vec + word_vectors[temp[i],]
         }
    }
    return(sent_vec/valid_words)
    
  }
  else if(method == "squared")
  {
    temp <- tokenize_words(sent, lowercase = T, simplify = T, stopwords = stopwords("en"))
    sent_vec = rep(0, 50)
    valid_words = 0
    for(i in 1:length(temp))
    {
      if(temp[i] %in% rownames(word_vectors))
      {
        valid_words = valid_words + 1
        sent_vec = sent_vec + (word_vectors[temp[i],])^2
      }
    }
    return(sent_vec/valid_words)
  }
  else if(method == "var")
  {
    tt = Sentence_Vectorizer(sent, method = "avg", word_vectors)
    temp <- tokenize_words(sent, lowercase = T, simplify = T, stopwords = stopwords("en"))
    sent_vec = rep(0, 50)
    valid_words = 0
    for(i in 1:length(temp))
    {
      if(temp[i] %in% rownames(word_vectors))
      {
        valid_words = valid_words + 1
        sent_vec = sent_vec + (tt - word_vectors[temp[i],])^2
      }
    }
    return(sent_vec/valid_words)
  }
}
Data_Base_Maker <- function(data, method = 'avg', word_vectors)
{
  pb <- txtProgressBar(max = length(data), style = 3)
  newdata = matrix(0,nrow = length(data), ncol = 50)
  for(i in 1:length(data))
  {
    temp <- as.numeric(Sentence_Vectorizer(data[i], method = method, word_vectors= word_vectors))
    newdata[i,] = temp
    setTxtProgressBar(pb, i)
  }
  return(as.data.frame(newdata))
}
Large_Word_Vectors <- read.table("~/Redditbot/glove.6B/glove.6B.50d.txt", sep = " ", header = FALSE,
                   quote = NULL, comment.char = "", row.names = 1,
                   nrows = -1)

One_Hot_Maker <- function(outputs)
{
  fac <- factor(outputs, labels = c(seq(1, length(unique(outputs)))))
  y_labs <- matrix(0, nrow = length(outputs), ncol = length(unique(outputs)))
  for(i in 1:length(outputs))
  {
    y_labs[i,as.numeric(fac[i])] = 1
    
  }
  return(y_labs)
}

askscience_Data <- askscience_Data[-c(which(askscience_Data$tag == 'meta'), which(is.na(askscience_Data$tag))),]

word_vectors <- Create_Word_Vectors(c(askscience_Data$Title, layer1data$title), seq(1,length(askscience_Data$Title) + length(layer1data$title), 1))
Train_my_emb <- Data_Base_Maker(askscience_Data$Title, 'avg', word_vectors)
Train_glove_emb <- Data_Base_Maker(askscience_Data$Title, 'avg', Large_Word_Vectors)
Codes_done("Done", "Word Vectors trained")

#### Train and Test creation ####
set.seed(69)
dat_my <- cbind(Train_my_emb, askscience_Data$tag) %>%
  Cross_val_maker(.2)
train_my <- dat_my$Train[,-51]
y_train_my <- One_Hot_Maker(dat_my$Train[,51])
test_my <- dat_my$Test
y_test_my <- One_Hot_Maker(dat_my$Test[,51])

dat_glove <- cbind(Train_glove_emb, askscience_Data$tag) %>%
  Cross_val_maker(.2)
train_glove <- dat_glove$Train[,-51]
y_train_glove <- One_Hot_Maker(dat_glove$Train[,51])
test_glove <- dat_glove$Test
y_test_glove <- One_Hot_Maker(dat_glove$Test[,51])








########### tests ########

tt <- (word_vectors["i", ] + word_vectors["love", ] + word_vectors["physics", ])/3
tt == Sentence_Vectorizer('nasd love physics', method = 'avg', word_vectors)

tt2 <- (word_vectors["i", ]^2 + word_vectors["love", ]^2 + word_vectors["physics", ]^2)/3
tt2 == Sentence_Vectorizer('i love physics', method = 'squared', word_vectors)

tt3 <-((tt - word_vectors["i", ])^2 + (tt - word_vectors["love", ])^2 + (tt - word_vectors["physics", ])^2)/3
tt3 == Sentence_Vectorizer('i love physics', method = 'var', word_vectors)
