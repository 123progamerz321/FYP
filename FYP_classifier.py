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
current = current.reset_index(drop=True)
former = former.reset_index(drop=True)

print (current.shape)
print (former.shape)

# Split training and testing set (ratio 8:2)
currentTrain = current[0: int(current.shape[0] * 0.8)]
currentTest = current[int(current.shape[0]*0.8):] 
formerTrain = former[0: int(former.shape[0] * 0.8)]
formerTest = former[int(former.shape[0]*0.8):] 
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
#    if count == 0:
#        print (text)
        
#    stopWords = set(stopwords.words("english"))
#    text = [i for i in text if i not in stopWords]
#    text = " ".join(text)
#        
#    text = text.split()
#    stemmer = SnowballStemmer('english')
#    stemmed_words = [stemmer.stem(word) for word in text]
#    if count == 0:
#        print (stemmed_words)
#    text = " ".join(stemmed_words)
    return text

#i = 0
currentPosReviews = []
currentNegReviews = []
formerPosReviews = []
formerNegReviews = []
#for rev in current["pros"]:
#    currentPosReviews.append(preProcessing(rev))

for i in range(current.shape[0]):
    if i < former.shape[0]:
        formerPosReviews.append(preProcessing(former["pros"][i]))
        formerNegReviews.append(preProcessing(former["cons"][i]))
    currentPosReviews.append(preProcessing(current["pros"][i]))
    currentNegReviews.append(preProcessing(current["cons"][i]))
    
currentPosReviews = currentPosReviews[0:3]
print (currentPosReviews)
tokenizer = Tokenizer(num_words=max_features)

    

