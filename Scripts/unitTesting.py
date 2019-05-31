# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:20:54 2019

@author: user
"""

import unittest

from function import preProcessing, plot_2d_space, imbalanced_resampling
from function import loadEmbedding, createEmbeddingMatrix, modelHistory, plot_cf
import os

#Test for removing non-alphabetic characters, digits, tab, space
class unitTesting(unittest.TestCase):
#    def test_removeNonAlphabetics(self):
#        self.assertTrue(preProcessing("#@$&@#)($*@)") == "")
#        self.assertTrue(preProcessing("abc#@$&@#") == "abc")
#        self.assertTrue(preProcessing("") == "")
#                                      
#    def test_removeDigits(self):
#        self.assertTrue(preProcessing("123456") == "")
#        self.assertTrue(preProcessing("asd12bcd") == "asdbcd")
#        self.assertTrue(preProcessing("") == "")
#        
#    def test_removeBoth(self):
#        self.assertTrue(preProcessing("1@#$$abc23456") == "abc")
#        self.assertTrue(preProcessing("asd12bcd*@#") == "asdbcd")
#        self.assertTrue(preProcessing(" ") == "")
#        self.assertTrue(preProcessing("  ") == "")
#        self.assertTrue(preProcessing("abcdef") == "abcdef")

#    def test_models_tokenizer_vectorizer_are_saved(self):
#        if (os.path.isfile("tokenizer.pickle")):
#            os.remove("tokenizer.pickle")
#
#        if (os.path.isfile("tfidfVectorizer.pickle")):
#            os.remove("tfidfVectorizer.pickle")
#                
#        if (os.path.isfile("review_model_gpu.json")):
#            os.remove("review_model_gpu.json")
#            
#        if (os.path.isfile("review_model_gpu.h5")):
#            os.remove("review_model_gpu.h5")
#            
#        if (os.path.isfile("svm_model_rev.sav")):
#            os.remove("svm_model_rev.sav")
#            
#        import nnModel
#        import svmModel
#        
#        self.assertTrue(os.path.isfile("tokenizer.pickle"))
#        self.assertTrue(os.path.isfile("tfidfVectorizer.pickle"))
#        self.assertTrue(os.path.isfile("review_model_gpu.json"))    
#        self.assertTrue(os.path.isfile("review_model_gpu.h5"))
#        self.assertTrue(os.path.isfile("svm_model_rev.sav"))
    
#    def test_convert_to_TFIDF(self):
#        import pandas as pd
#        from sklearn.feature_extraction.text import TfidfVectorizer
#        numOfRev = 2
#        numOfFeatures = 9 # Boss, is, very, nice, salary, high, working, environment, good
#        data = ["Boss is very nice, salary is very high", "working environment is very good"]
#        data = pd.Series(data)
#        TFIDF_vectorizer = TfidfVectorizer()
#        feature_vector = TFIDF_vectorizer.fit_transform(data)
#        self.assertEqual(feature_vector.shape[0], numOfRev)
#        self.assertEqual(feature_vector.shape[1], numOfFeatures)
    
    def test_convert_to_TFIDF(self):
        import pickle
        import pandas as pd
        import numpy as np
        from keras.models import model_from_json
        from keras.preprocessing.sequence import pad_sequences
        revModel = "review_model_gpu.json"
        revModelWeights = "review_model_gpu.h5"
        
        # load json and create model (review predicts rating)
        json_file = open(revModel, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        review_model = model_from_json(loaded_model_json)
        
        review_model.load_weights(revModelWeights)
        review_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        pros_rev = input("Enter a pros review: ")
        cons_rev = input("Enter a cons review: ")
        combine_rev = preProcessing(pros_rev + " " + cons_rev)
        combine_rev = pd.Series(combine_rev)
        
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)        
            
        maxlen = 200
        tokenized_rev = tokenizer.texts_to_sequences(combine_rev)
        user_rev = pad_sequences(tokenized_rev, maxlen=maxlen, padding='post', truncating='post')
        
        # Predict rating based on user review (LSTM-CNN)
        model_pred = review_model.predict([user_rev], batch_size=1024, verbose=1)
        print ("LSTM-CNN Overall Rating:", np.argmax(model_pred[0]))
        
        self.assertEqual(type(np.argmax(model_pred[0])) == str, True)
        self.assertEqual(type(np.argmax(model_pred[0])) == bool, False)
        self.assertEqual(type(np.argmax(model_pred[0])) == int, False)
    
if __name__ == '__main__':
    unittest.main()
    
