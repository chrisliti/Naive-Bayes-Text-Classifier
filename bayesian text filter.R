#Reading in the data
sms_raw <- read.csv("C:/Users/Chris Liti/Desktop/R Data/sms_spam.csv")

str(sms_raw)

#Convert type to character
sms_raw$type <- factor(sms_raw$type)

#Check out the distribution of type
table(sms_raw$type)

#Load tm package for cleaning and processing the data
library(tm)

#Create a corpus(collection of text docs)
#Vcorpus is stored in memory instead of disk

sms_corpus <- VCorpus(VectorSource(sms_raw$text))

inspect(sms_corpus[1:2])

#To view actual message
as.character(sms_corpus[[1]])

#To view several actual messages use lapply
lapply(sms_corpus[1:3],as.character)

#Applying Transformations
#To lower case characters

sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
as.character(sms_corpus_clean[[1])

#To lower
#fxns created in tm package don't need to be wrapped
sms_corpus_clean <- tm_map(sms_corpus_clean,removeNumbers)

#Remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords,stopwords())

#Remove punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean,removePunctuation)

#require snowballc package for stemming
library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean,stemDocument)

#Remove whitespaces
sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)

#Tokenization: Splitting messages into individual contents
#DTM(Document Term Matrix): Rows indicate documents and columns indicate terms

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

#Split into training and testing sets
sms_dtm_train <- sms_dtm[1:4180,]
sms_dtm_test <- sms_dtm[4181:5574,]

#Aquire the labels
names(sms_raw)
sms_train_labels <- sms_raw[1:4180,"type"]
sms_test_labels <- sms_raw[4181:5574,"type"]

#Test sample propotions
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#Visualizing using wordclouds
library(wordcloud)
wordcloud(sms_corpus_clean,min.freq = 50,random.order = F)

#Visualizing both spam and ham texts
spam <- subset(sms_raw,type=="spam")
ham <- subset(sms_raw,type=="ham")

wordcloud(spam$text,scale=c(3,0.5),max.words = 40)
wordcloud(ham$text,scale=c(3,0.5),max.words = 40)

#TO reduce features omit words that appear in less than 5 texts
sms_freq_words <- findFreqTerms(sms_dtm_train,5)

sms_dtm_freq_train <- sms_dtm_train[,sms_freq_words]
sms_dtm_freq_test<- sms_dtm_test[,sms_freq_words]
dim(sms_dtm_freq_train)

#Convert the counts into categorical variables since naive bayes classifier works well on categorical variables
#Create a fxn

convert_counts <- function(x){
  x <- ifelse(x>0,"Yes","No")
}

#Applt it on the DTMS
sms_train <- apply(sms_dtm_freq_train,2,convert_counts)
sms_test <- apply(sms_dtm_freq_test,2,convert_counts)

View(sms_train)

#Training model 
library(e1071)

sms_classifier <- naiveBayes(sms_train,sms_train_labels)

#Make Predictions
sms_test_pred <- predict(sms_classifier,sms_test)

#cross tabulate
library(gmodels)
CrossTable(sms_test_pred,sms_test_labels,prop.chisq = FALSE,prop.t = FALSE,prop.r=FALSE,dnn = c("Predicted","Actual"))
mean(sms_test_pred==sms_test_labels)

#Improve Model
#The Laplace estimator essentially adds a small number to each of the counts in the frequency
#table, which ensures that each feature has a nonzero probability of occurring with
#each class
sms_classifier2 <- naiveBayes(sms_train,sms_train_labels,laplace = 1)
#Make Predictions
sms_test_pred2 <- predict(sms_classifier2,sms_test)

#cross tabulate
library(gmodels)
CrossTable(sms_test_pred2,sms_test_labels,prop.chisq = FALSE,prop.t = FALSE,prop.r=FALSE,dnn = c("Predicted","Actual"))
mean(sms_test_pred2==sms_test_labels)

#There has been loss in general accuracy and less ham predicted as spam which is a better alternative
