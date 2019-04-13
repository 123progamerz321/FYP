# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:53:19 2019

@author: user
"""

import pandas as pd
import numpy as np 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

path = "../Desktop/reviews.csv"
data = pd.read_csv(path)
reviews = data[["job-title", "pros", "cons"]]
reviews = reviews.rename(columns = {"job-title": "employeeType"})

#Drop rows with that contain empty cell/cells
reviews = reviews.dropna() 

#Only need to know whether those reviewers are current/former employees, we don't care about job positions of them
reviews["employeeType"] = reviews["employeeType"].apply(lambda x: "Current Employee" if x[0:16] == "Current Employee" 
                                                        else "Former Employee")

# To see if there exists some cells with different value (neither current nor former employee)
print (reviews.shape)
print (reviews["employeeType"].value_counts())

# Separate current employee reviews and former employee reviews
current = reviews[reviews["employeeType"] == "Current Employee"]
former = reviews[reviews["employeeType"] == "Former Employee"]

# Reset index
#current = current.reset_index(drop=True)
#former = former.reset_index(drop=True)

#print (current.shape)
#print (former.shape)

# Split training and testing set (ratio 8:2) and reset_index
currentTrain = current[0: int(current.shape[0] * 0.8)].reset_index(drop=True)
currentTest = current[int(current.shape[0]*0.8):].reset_index(drop=True)
formerTrain = former[0: int(former.shape[0] * 0.8)].reset_index(drop=True)
formerTest = former[int(former.shape[0]*0.8):].reset_index(drop=True) 
#print (currentTrain)
#print (currentTest)
#print (formerTrain)
#print (formerTest)

# Create and compile regular expression patterns into regular expression objects
# specialCharRemoval for removing non-digit and non-alphabetical characters
# replaceDigit for replacing digit
# re.IGNORECASE perform case-insensitive matching; expressions like [a-z] will also match uppercase letters
specialCharRemoval = re.compile(r'[^a-z\d ]', re.IGNORECASE)
replaceDigit = re.compile(r'\d+', re.IGNORECASE)

def preProcessing(text):
    text = specialCharRemoval.sub('', text)
    text = replaceDigit.sub('', text)
    text = " ".join(text.lower().split())
        
#    stopWords = set(stopwords.words("english"))
#    text = [i for i in text if i not in stopWords]
#    text = " ".join(text)
#        
#    text = text.split()
#    stemmer = SnowballStemmer('english')
#    stemmed_words = [stemmer.stem(word) for word in text]
#    text = " ".join(stemmed_words)
    return text

currentPosReviews_train, currentNegReviews_train = [], []
formerPosReviews_train, formerNegReviews_train = [], []

currentPosReviews_test, currentNegReviews_test = [], []
formerPosReviews_test, formerNegReviews_test = [], []


for i in range(currentTrain.shape[0]):
    if i < formerTrain.shape[0]:
        formerPosReviews_train.append(preProcessing(formerTrain["pros"][i]))
        formerNegReviews_train.append(preProcessing(formerTrain["cons"][i]))
    currentPosReviews_train.append(preProcessing(currentTrain["pros"][i]))
    currentNegReviews_train.append(preProcessing(currentTrain["cons"][i]))
    
for e in range(currentTest.shape[0]):
    if e < formerTest.shape[0]:
        formerPosReviews_test.append(preProcessing(formerTest["pros"][e]))
        formerNegReviews_test.append(preProcessing(formerTest["cons"][e]))
    currentPosReviews_test.append(preProcessing(currentTrain["pros"][e]))
    currentNegReviews_test.append(preProcessing(currentTrain["cons"][e]))

# keras parameters setup
embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)

# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length 
#(with truncation or padding as needed).
cTokenizer = Tokenizer(num_words=max_features)
fTokenizer = Tokenizer(num_words=max_features)
cTokenizer.fit_on_texts(currentPosReviews_train + currentNegReviews_train)
fTokenizer.fit_on_texts(formerPosReviews_train + formerNegReviews_train)

################################ CODE BELOW THIS LINE ONLY FOR CURRENT EMPLOYEE MODEL ################################ 
list_tokenized_train = cTokenizer.texts_to_sequences(currentPosReviews_train + currentNegReviews_train)
list_tokenized_test = cTokenizer.texts_to_sequences(currentPosReviews_test + currentNegReviews_test)

## Tokenization testing
## word_counts: A dictionary of words and their counts.
## word_docs: A dictionary of words and how many documents each appeared in.
## word_index: A dictionary of words and their uniquely assigned integers.
## document_count:An integer count of the total number of documents that were used to fit the Tokenizer.
#print(cTokenizer.word_counts)
#print(cTokenizer.word_docs)
#print(cTokenizer.word_index["if"])
#print(cTokenizer.document_count)

# Each indexed-reviews has different length, we have to feed data that has a fixed number of features (consistent length)
# Find best "maxlen" to set. If we put it too short, we might lose some useful feature that could cost us some accuracy points down the path.
# If set it too long, our LSTM cell will have to be larger to store the possible values or states.
# Code below plots a histogram showing the distribution of the length of the reviews in our training dataset (Current employee)
totalNumWords = [len(review) for review in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
plt.show()

# As we can see most of the review length is about 30+, but just in case we set the maxlen to 200
# Use padding to make shorter reviews as long as the others by filling zeros.
# Use padding to trim longer reviews to the maxlen specified. 
maxlen = 200 # max number of words in a comment to use
CX_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
CX_test = pad_sequences(list_tokenized_test, maxlen=maxlen)




    

