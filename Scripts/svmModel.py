# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:27:34 2019

@author: ASUS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from function import preProcessing, plot_cf


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read csv file and extract specific columns that we want to work on
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
path = "C:/Users/A/Desktop/FYP/reviews.csv"
data = pd.read_csv(path)
rev = data[["pros", "cons", "overall-ratings", "work-balance-stars", "culture-values-stars", "carrer-opportunities-stars", "comp-benefit-stars", "senior-mangemnet-stars"]]
rev = rev.rename(columns = {"overall-ratings": "rating", "work-balance-stars": "work_balance", "culture-values-stars": "culture_val", 
                            "carrer-opportunities-stars": "career_opp", "comp-benefit-stars": "comp_benefit", "senior-mangemnet-stars": "senior_mng"})

rev = rev.dropna() 
rev = rev.drop(rev[(rev.work_balance == "none") | (rev.culture_val == "none") | (rev.career_opp == "none") | (rev.comp_benefit == "none") | 
                   (rev.senior_mng == "none")].index)

x = rev[["pros", "cons", "work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
y = rev["rating"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 100)
x_train = x_train.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data preprocessing for predicting test data's overall rating (extract relevant columns)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rate = x_train[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]
x_test_rate = x_test[["work_balance", "culture_val", "career_opp", "comp_benefit", "senior_mng"]]

x_train_rate = x_train_rate.values.astype(float)
x_test_rate = x_test_rate.values.astype(float)
y_train_rate = y_train.to_numpy()
y_test_rate = y_test.to_numpy()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Build SVM classifier that use other ratings to predict overall ratings 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
SVM_linear = LinearSVC(C=0.1)
SVM_linear.fit(x_train_rate, y_train_rate)
pred = SVM_linear.predict(x_test_rate)
#plot_cf(y_test_rate, pred, classifier = "SVM", normalize = True)
#plt.show()
#print("\n" + "Model accuracy (ratings predict overall rating)", ":", accuracy_score(y_test_rate, pred))
#print('Classification Report: ' + "SVM" + "\n")
#print(classification_report(y_test_rate, pred))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Save model (ratings predict overall rating) to disk
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filename = 'svm_model_rat.sav'
pickle.dump(SVM_linear, open(filename, 'wb'))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Choosing different subsets. line 85-87: Concatenating both positive and negative sentiment reviews
Must comment out line 85-87 if want to try different subsets
Feel free to uncomment line 89-91 or line 93-95 to see the model performance on different subsets
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = x_train[["pros", "cons"]]
x_test_rev = x_test[["pros", "cons"]]

# Combine pros and cons reviews
x_train_rev["review"] = x_train_rev[["pros", "cons"]].apply(lambda x: ". ".join(x), axis = 1)
x_test_rev["review"] = x_test_rev[["pros", "cons"]].apply(lambda x: ". ".join(x), axis = 1)

## Pros only
#x_train_rev["review"] = x_train_rev[["pros"]]
#x_test_rev["review"] = x_test_rev[["pros"]]

## Cons only
#x_train_rev["review"] = x_train_rev["cons"]
#x_test_rev["review"] = x_test_rev["cons"]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show distributions of ratings
Feel free to uncomment the code to see the plot
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#fig = plt.figure(figsize=(8,6))
#rev.groupby("ratings").rev.count().plot.bar(ylim=0)
#plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data preprocessing for predicting test data's overall rating  
(remove non-alphabetic characters, remove emoticons, remove digit, TFIDF)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = x_train_rev["review"]
x_test_rev = x_test_rev["review"]
x_train_rev = x_train_rev.apply(lambda x: preProcessing(x)).reset_index(drop=True)
x_test_rev = x_test_rev.apply(lambda x: preProcessing(x)).reset_index(drop=True)

#TFIDF Vectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
X_train_tfidf = TFIDF_vectorizer.fit_transform(x_train_rev)
#print(X_train.head())
#print(TFIDF_vectorizer.vocabulary_)
#print(X_train_tfidf)
#print(X_train_tfidf.toarray())
#print(TFIDF_vectorizer.transform([X_train[0]]).toarray())
#X_train[0]

# Save tfidfVectorizer object
with open('tfidfVectorizer.pickle', 'wb') as handle:
    pickle.dump(TFIDF_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_test_tfidf = TFIDF_vectorizer.transform(x_test_rev)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Build SVM classifier that use reviews to predict overall ratings 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#SVM_linear = svm.SVC(kernel='linear')

#hyperparameters tuning
#from sklearn.model_selection import GridSearchCV
#def svc_param_selection(X, y, nfolds):
#    Cs = {'C': [0.001, 0.10, 0.1, 10, 25, 50]}
#    grid_search = GridSearchCV(LinearSVC(), Cs, cv=nfolds)
#    grid_search.fit(X, y)
#    grid_search.best_params_
#    return grid_search.best_params_
#params = svc_param_selection(X_train_tfidf, y_train, 5)
#best params is C = 0.1

SVM_linear.fit(X_train_tfidf, y_train)
prediction_SVM_linear = SVM_linear.predict(X_test_tfidf)
print ("Model accuracy (review predict overall rating):", accuracy_score(y_test, prediction_SVM_linear))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot confusion matrix
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
plot_cf(y_test, prediction_SVM_linear, "SVM", normalize = True)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classification report (precision, recall, f1, micro&macro avg, weighted avg)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
report = classification_report(y_test, prediction_SVM_linear)
print(report)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Save SVM model to disk
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filename = 'svm_model_rev.sav'
pickle.dump(SVM_linear, open(filename, 'wb'))
