# Pretreatment 
library(dplyr)
library(forcats)
library(text2vec)
library(tm) # For stemming
StopWordRm = function(sent, stopwords = stopwords){
  temp = function(x, y = stopwords){
    if((x %in% y)){
      return("")
    }
    else(return(x))
  }
  sent = strsplit(sent, ' ', fixed = TRUE)[[1]]
  sent = paste(lapply(sent, temp, stopwords), collapse = " ")
  return(sent)
}
StopWordMaker = function(titles, cutoff = 20){
  test = unlist(lapply(as.vector(titles), strsplit, split = ' ', fixed = TRUE))
  stopwords = test %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(cutoff) %>%
    names()
  return(stopwords)
}
askscience_Data <- read.csv("~/Documents/GitHub/Redditbot/askscience_Data.csv")
subs = c("physics", "bio", "med", "geo", "chem", "astro", "eng")
askscience_Data$tag = askscience_Data$tag %>%
  fct_collapse("Other" = c(as.character(unique(askscience_Data$tag)[which(!(unique(askscience_Data$tag) %in% subs))])))

dat = LilRhino::Cross_val_maker(askscience_Data, .15)


titles = as.character(dat$Train$Title) %>%
  lapply(gsub, pattern = "[^[:alnum:][:space:]]",replacement = "") %>%
  lapply(stemDocument) %>%
  lapply(tolower)
stopwords = StopWordMaker(titles, 20)



it   = itoken(unlist(titles), tolower, word_tokenizer, ids = tags)
vocab = create_vocabulary(it, stopwords = stopwords)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 5)
glove$fit_transform(tcm, n_iter = 20)
word_vecs = glove$components

# When you pick this back up, write a function for sentence vector averaging
# and for turning your data into sentence vectors! 


## 0) Pretreatment

Pretreatment = function(title_vec){
  titles = as.character(title_vec) %>%
    lapply(gsub, pattern = "[^[:alnum:][:space:]]",replacement = "") %>%
    lapply(stemDocument) %>%
    lapply(tolower)
  return(titles)
}

## 1) Stopword maker


StopWordMaker = function(titles, cutoff = 20){
  test = unlist(lapply(as.vector(titles), strsplit, split = ' ', fixed = TRUE))
  stopwords = test %>%
    table() %>%
    sort(decreasing = TRUE) %>%
    head(cutoff) %>%
    names()
  return(stopwords)
}


## 2) Auto-creating word matrix 


Embedding_Matrix = function(words, vocab_min, stopwords, skip_gram, vector_size){
  it   = itoken(unlist(words), tolower, word_tokenizer)
  vocab = create_vocabulary(it, stopwords = stopwords)
  vocab <- prune_vocabulary(vocab, term_count_min = vocab_min)
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
  # use window of 5 for context words
  tcm <- create_tcm(it, vectorizer, skip_grams_window = skip_gram)
  glove = GlobalVectors$new(word_vectors_size = vector_size, vocabulary = vocab, x_max = 5)
  glove$fit_transform(tcm, n_iter = 20)
  return(glove$components)
}
colnames(temp)

## 3) Sentence Converter

###### 3a) word puller
Vector_puller = function(word, emb_matrix){
  if(word %in% rownames(emb_matrix)){
    return(emb_matrix[word,])
  }
  else{
    return(rep(0, ncol(emb_matrix)))
  }
}

Sentence_Vector = function(sentence, emb_matrix, stopwords){
  words = strsplit(sentence, " ", fixed = TRUE)[[1]]
#  for(i in 1:4){
#    Vector_puller(word[i], emb_matrix)
#    
#  }
  vec = lapply(words, Vector_puller, emb_matrix)
  df <- data.frame(matrix(unlist(vec), nrow=ncol(emb_matrix), byrow=F))
  return(rowSums(df)/length(unique(colSums(df))))
}









