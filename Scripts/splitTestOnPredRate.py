
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:58:08 2019

@author: A
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Conv1D, MaxPooling1D, GRU, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, Concatenate
from keras.models import Model
from keras.utils import to_categorical

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from function import preProcessing, plot_2d_space, imbalanced_resampling
from function import loadEmbedding, createEmbeddingMatrix, modelHistory, plot_cf

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read csv file and extract specific columns that we want to work on
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
path = "C:/Users/A/Desktop/FYP/reviews.csv"
data = pd.read_csv(path)
rev = data[["pros", "cons", "overall-ratings", "work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]]
rev = rev.rename(columns = {"overall-ratings": "rating", "work-balance-stars": "work_balance", "culture-values-stars": "culture_val", 
                            "carrer-opportunities-stars": "career_opp", "comp-benefit-stars": "comp_benefit", "senior-mangemnet-stars": "senior_mng"})

# Drop rows with missing values in any of the five aspects rating 
rev = rev.dropna()
rev = rev.drop(rev[(rev.work_balance == "none") | (rev.culture_val == "none") | (rev.career_opp == "none") | (rev.comp_benefit == "none") | 
                   (rev.senior_mng == "none")].index)

x = rev[["pros", "cons", "work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
y = rev["rating"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train = x_train.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

# array created for resampling purpose (data imbalanced) and one hot encoding
ytrain_arr = np.array(y_train)
ytest_arr = np.array(y_test)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data preprocessing for predicting test data's overall rating (extract relevant columns, reshape, one hot encode)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rate = x_train[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
x_test_rate = x_test[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]

# Convert dataframe to numpy array
x_train_rate = x_train_rate.values.astype(float)
x_test_rate = x_test_rate.values.astype(float)

# Reshape to 3d tensor (numple of samples, timesteps, features)
x_train_rate = x_train_rate.reshape(-1, 1, x_train_rate.shape[1])
x_test_rate = x_test_rate.reshape(-1, 1, x_test_rate.shape[1])

# One hot encode y
y_train_rate = to_categorical(ytrain_arr)
y_test_rate = to_categorical(ytest_arr)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Bidirectional LSTM with two fully connected layers. We add some dropout to the LSTM since even 2 epochs is enough to overfit.
This model takes the 5 aspects rating as inputs and used them to predict the overall ratings
Comment out CudNNLSTM and uncomment out regular LSTM if gpu is not available
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# shape: A shape tuple (integer), not including the batch size. 
# For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
# In our case, input tensor for sequences of 1 timestep, each containing a 5-dimensional vector

# Only have 5 aspects rating
maxlen = 5
batch_size = 128
epochs = 20
val_split = 0.1
inp = Input(shape=(1, maxlen))

# input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: int >= 0. Dimension of the dense embedding.
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inp)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(y_train_rate.shape[-1], activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary() 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train_rate, y_train_rate, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks = [early_stopping]);

rating_pred_test = model.predict([x_test_rate], batch_size=1024, verbose=1)
rating_pred_val = model.predict([x_train_rate], batch_size=1024, verbose=1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Feel free to uncomment out the following code (line 115-155) to check the performance measures used to evaluate this model
We also include traditional machine learning classifiers (logistic regression, naive bayes, svm, random forest) for comparison
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#acc = sum([np.argmax(y_test_rate[i])==np.argmax(rating_pred_test[i]) for i in range(x_test_rate.shape[0])])/x_test_rate.shape[0]
#print ("Model accuracy:", acc)
#
## Plot confusion matrix
#y_pred_copy = np.copy(rating_pred_test)
#for row in range(y_pred_copy.shape[0]):
#    y_pred_copy[row] = (y_pred_copy[row] == np.amax(y_pred_copy[row])).astype(int)
#    
#np.set_printoptions(precision=2)
#plot_cf(y_test_rate.argmax(axis=1), y_pred_copy.argmax(axis=1), "LSTM", normalize = True)
#plt.show()
#
## Classification report
#print('Classification Report')
#print(classification_report(y_test_rate.argmax(axis=1), rating_pred_test.argmax(axis=1)))
#    
#
## Traditional machine learning classifiers
## Including logistic regression, naive bayes, svm, random forest
#
## Created for traditional machine learning 
#ml_train = x_train[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
#ml_test = x_test[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
#ml_xtrain = ml_train.values.astype(float)
#ml_xtest = ml_test.values.astype(float)
#ml_ytrain = y_train.to_numpy()
#ml_ytest = y_test.to_numpy()
#
#lr = LogisticRegression(solver='lbfgs')
#mnb = MultinomialNB()
#gnb = GaussianNB()
#svc = LinearSVC(C=1.0)
#rfc = RandomForestClassifier(n_estimators=100)
#classifiers = [(lr, "Logistic_regression"), (mnb, 'Multinomial Naive Bayes'), (gnb, 'Gaussian Naive Bayes'), (svc, 'Support Vector Classification'), (rfc, 'Random Forest')]
#
#for clf, name in classifiers:
#    clf.fit(ml_xtrain, ml_ytrain)
#    pred = clf.predict(ml_xtest)
#    plot_cf(ml_ytest, pred, classifier = name, normalize = True)
#    plt.show()
#    print("\n" + name, ":", accuracy_score(ml_ytest, pred))
#    print('Classification Report: ' + name + "\n")
#    print(classification_report(ml_ytest, pred))
    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Convert into 1d array
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
rating_pred_test = rating_pred_test.argmax(axis = 1)
rating_pred_val = rating_pred_val.argmax(axis = 1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Extract pros and cons reviews
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = x_train[["pros", "cons"]]
x_test_rev = x_test[["pros", "cons"]]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Investigate which combination has the best performance measures in predicting ratings

The combination with the best performance is already uncommented, feel free to observe the performance of model
on different combination by uncommenting the selected combination and commenting out other combinations

Assuming ratings 1,2,3 = cons; rating 4 = combine, ratings 5 is our current combination
Following is the regular for loop of the list comprehension:

review = []
for i in x_train_rev:
    if i < x_train_rev.shape[0] * (1-validation_split):
        if y_train[i] == 5:
            review.append(x_train_rev["pros"][i])
        elif y_train[i] == 4:
            review.append(x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i])
        else:
            review.append(x_train_rev["cons"][i])
            
    else (if we reached validation data):
        if rating_pred_val[i] == 5:
            review.append(x_train_rev["pros"][i])
        elif rating_pred_val[i] == 4:
            review.append(x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i])
        else:
            review.append(x_train_rev["cons"][i])
            
x_train_rev["review"] = review

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
val_split = 0.1

# Ratings 1,2,3 = cons; rating 4 = combine, ratings 5 = pros
x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 else x_train_rev["cons"][i]) 
                        if i < x_train_rev.shape[0] * (1-val_split) 
                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 else x_train_rev["cons"][i]) 
                        for i in x_train_rev.index]

x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4
                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1 = cons; rating 2,3,4 = combine, ratings 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 3 or y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 3 or rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 3 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1 = cons; rating 2 = combine, ratings 3,4,5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 4 or y_train[i] == 3 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 4 or rating_pred_val[i] == 3 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 3 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1,2 = cons; rating 3,4 = combine, ratings 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 3 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 3 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 3
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1 = cons; rating 2,3 = combine, ratings 4,5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 4 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 3 or y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 4 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 3 or rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5 or rating_pred_test[i] == 4
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 3 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1,2 = cons; rating 3 = combine, ratings 4,5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 4 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 3 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 4 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 3 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5 or rating_pred_test[i] == 4
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 3
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1,3 = cons; rating 2, 4 = combine, ratings 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 3 = cons; rating 1, 2, 4 = combine, ratings 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 2 or y_train[i] == 1 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 2 or rating_pred_val[i] == 1 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 2 or rating_pred_test[i] == 1
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 3 = cons; rating 2, 4 = combine, ratings 1, 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 1 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 1 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5 or rating_pred_test[i] == 1
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 3 = cons; rating 1, 4 = combine, ratings 2, 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 2 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 1 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 2 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 1 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5 or rating_pred_test[i] == 2
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 1
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


## Ratings 1 = cons; rating 2, 4 = combine, ratings 3, 5 = pros
#x_train_rev["review"] = [(x_train_rev["pros"][i] if y_train[i] == 5 or y_train[i] == 3 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if y_train[i] == 4 or y_train[i] == 2 else x_train_rev["cons"][i]) 
#                        if i < x_train_rev.shape[0] * (1-val_split) 
#                        else (x_train_rev["pros"][i] if rating_pred_val[i] == 5 or rating_pred_val[i] == 3 else x_train_rev["pros"][i] + ". " + x_train_rev["cons"][i] if rating_pred_val[i] == 4 or rating_pred_val[i] == 2 else x_train_rev["cons"][i]) 
#                        for i in x_train_rev.index]
#
#x_test_rev["review"] = [x_test_rev["pros"][i] if rating_pred_test[i] == 5 or rating_pred_test[i] == 3
#                        else x_test_rev["pros"][i] + ". " + x_test_rev["cons"][i] if rating_pred_test[i] == 4 or rating_pred_test[i] == 2
#                        else x_test_rev["cons"][i] for i in x_test_rev.index]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data preprocessing (remove emoticons, remove non-alphabetic characters, remove digit, 
                    one hot encode labels, tokenization)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = x_train_rev["review"]
x_test_rev = x_test_rev["review"]

y_train_val = y_train.value_counts()
y_test_val = y_test.value_counts()
    
x_train_rev = x_train_rev.apply(lambda x: preProcessing(x)).reset_index(drop=True)
x_test_rev = x_test_rev.apply(lambda x: preProcessing(x)).reset_index(drop=True)

# One hot encode y
y_train_rev = to_categorical(ytrain_arr)
y_test_rev = to_categorical(ytest_arr)

max_features = 20000
maxlen = 200
embed_size = 300
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train_rev))
tokenized_train = tokenizer.texts_to_sequences(x_train_rev)
tokenized_test = tokenizer.texts_to_sequences(x_test_rev)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Each indexed-x_train_rev has different length, we have to feed data that has a fixed number of features (consistent length)
Find best "maxlen" to set. If we put it too short, we might lose some useful feature that could cost us some accuracy points down the path.
If set it too long, our LSTM cell will have to be larger to store the possible values or states.
Code below plots a histogram showing the distribution of the length of the x_train_rev in our training dataset (Current employee)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#totalNumWords = [len(review) for review in tokenized_train]
#plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
#plt.xlabel('Review length')
#plt.ylabel('Frequency')
#plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
As we can see most of the review length is about 30+, but just in case we set the maxlen to 200
Use padding to make shorter x_train_rev as long as the others by filling zeros.
Use padding to trim longer x_train_rev to the maxlen specified. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = pad_sequences(tokenized_train, maxlen=maxlen, padding='post', truncating='post') # Can play with padding and truncating parameters (default padding & truncating = "pre")
x_test_rev = pad_sequences(tokenized_test, maxlen=maxlen, padding='post', truncating='post')  # Set truncating = "pos" removes values at the end of the sequences larger than maxlen


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Scatter plot for visualizing our data points in a 2d space
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#plot_2d_space(x_train_rev, ytrain_arr)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read pre-trained word vectors into a dictionary from word->vector.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
embeddings_index = loadEmbedding("glove")
#embeddings_index = loadEmbedding("fasttext")
#embeddings_index = loadEmbedding("word2vec")
        
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. 
We'll use the same mean and stdev of embeddings when generating the random init.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
embedding_matrix = createEmbeddingMatrix(max_features, embed_size, embeddings_index, tokenizer)
            

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Bidirectional LSTM + 1DConvolution with two fully connected layers. 
We add some dropout to the LSTM, after globalMaxPooling and after first dense layer as even 2 epochs is enough to overfit.
Comment out CudNNLSTM and uncomment out regular LSTM if GPU is not available
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# shape: A shape tuple (integer), not including the batch size. 
# For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.

inp = Input(shape=(maxlen,))
x = Embedding(min(max_features, len(tokenizer.word_index)), embed_size, weights=[embedding_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
conv1 = Conv1D(50, kernel_size = 3, activation = 'relu')(x)
conv2 = Conv1D(50, kernel_size = 4, activation = 'relu')(x)
conv3 = Conv1D(50, kernel_size = 5, activation = 'relu')(x)
conv1 = GlobalMaxPool1D()(conv1)
conv2 = GlobalMaxPool1D()(conv2)
conv3 = GlobalMaxPool1D()(conv3)
x = Concatenate()([conv1,conv2,conv3])
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(y_train_rev.shape[1], activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary() 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train_rev, y_train_rev, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks = [early_stopping]);

# summarize history for accuracy and loss
modelHistory(history)
y_pred = model.predict([x_test_rev], batch_size=1024, verbose=1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Testing accuracy
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
acc = sum([np.argmax(y_test_rev[i])==np.argmax(y_pred[i]) for i in range(x_test_rev.shape[0])])/x_test_rev.shape[0]
print ("Model accuracy:", acc)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot confusion matrix
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
y_pred_copy = np.copy(y_pred)
for row in range(y_pred_copy.shape[0]):
    y_pred_copy[row] = (y_pred_copy[row] == np.amax(y_pred_copy[row])).astype(int)

np.set_printoptions(precision=2)
plot_cf(y_test_rev.argmax(axis=1), y_pred_copy.argmax(axis=1), "LSTM", normalize = True)
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classification report (precision, recall, f1, micro&macro avg, weighted avg)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Classification Report')
print(classification_report(y_test_rev.argmax(axis=1), y_pred.argmax(axis=1)))
