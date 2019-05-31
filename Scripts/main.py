# -*- coding: utf-8 -*-
"""
Created on Thu May 23 02:05:03 2019

@author: A
"""
import pandas as pd
import numpy as np
import pickle
from keras.models import model_from_json
from function import preProcessing
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

if __name__ == "__main__":
    revModel = "review_model_gpu.json"
    revModelWeights = "review_model_gpu.h5"
    ratModel = "ratings_model_gpu.json"
    ratModelWeights = "ratings_model_gpu.h5"
    
    # If the running device does not have GPU, we run regular LSTM model
    if not tf.test.is_gpu_available():
        revModel = "review_model_cpu.json"
        revModelWeights = "review_model_cpu.h5"
        ratModel = "ratings_model_cpu.json"
        ratModelWeights = "ratings_model_cpu.h5"
        
    # load json and create model (review predicts rating)
    json_file = open(revModel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    review_model = model_from_json(loaded_model_json)
    
    # load json and create model (rating predicts rating)
    json_file = open(ratModel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    ratings_model = model_from_json(loaded_model_json)
    
    # load weights into new model and evaluate on user input (review predict ovreall rating)
    review_model.load_weights(revModelWeights)
    review_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    # load weights into new model and evaluate on user input (ratings predict overall rating)
    ratings_model.load_weights(ratModelWeights)
    ratings_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
    # load svm models
    svm_model_rev = pickle.load(open("svm_model_rev.sav", "rb"))
    svm_model_rat = pickle.load(open("svm_model_rat.sav", "rb"))
    
    # Ask user to enter pros and cons reviews or aspect rating
    print ()
    mode = input("Use review or aspect ratings to predict overall rating (r/a)? ")
    if mode.strip().lower() == "r":
        pros_rev = input("Enter pros review: ")
        cons_rev = input("Enter cons review: ")
        combine_rev = preProcessing(pros_rev + " " + cons_rev)
        combine_rev = pd.Series(combine_rev)
        
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)        
            
        maxlen = 200
        tokenized_rev = tokenizer.texts_to_sequences(combine_rev)
        user_rev = pad_sequences(tokenized_rev, maxlen=maxlen, padding='post', truncating='post')
        
        # SVM
        with open("tfidfVectorizer.pickle", "rb") as handle:
                vectorizer = pickle.load(handle)
        
        X_test_tfidf = vectorizer.transform(combine_rev)
        result = svm_model_rev.predict(X_test_tfidf)
        print ("\nSVM Overall Rating:", int(result[0]))
        
        # Predict rating based on user review (LSTM-CNN)
        model_pred = review_model.predict([user_rev], batch_size=1024, verbose=0)
        print ("LSTM-CNN Overall Rating:", np.argmax(model_pred[0]))
    
    elif mode.strip().lower() == "a":
        work_bal = input("Work balance: ")
        cul_val = input("Culture values: ")
        car_opp = input("Career opportunities: ")
        comp_ben = input("Company benefit: ")
        snr_mng = input("Senior management: ")
        
        user_inp = np.asarray([work_bal, cul_val, car_opp, comp_ben, snr_mng], dtype = np.float64)
        nn_inp = user_inp.reshape(-1, 1, 5)
        svm_inp = user_inp.reshape(-1, 5)
        
        # SVM
        result = svm_model_rat.predict(svm_inp)
        print ("\nSVM Overall Rating:", int(result[0]))
        
        # Predict rating based on other ratings(LSTM-CNN)
        model_pred = ratings_model.predict([nn_inp], batch_size=1024, verbose=0)
        print ("LSTM-CNN Overall Rating:", np.argmax(model_pred[0]))
        
    else:
        print ("Invalid input")
