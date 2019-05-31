
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
from keras.callbacks import EarlyStopping

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from function import preProcessing, plot_2d_space, imbalanced_resampling
from function import loadEmbedding, createEmbeddingMatrix, modelHistory, plot_cf

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read csv file and extract specific columns that we want to work on
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
path = "C:/Users/A/Desktop/FYP/reviews.csv"
data = pd.read_csv(path)
reviews = data[["company", "pros", "cons", "overall-ratings"]]
reviews = reviews.rename(columns = {"overall-ratings": "rating"})

# Drop rows with missing values
reviews = reviews.dropna()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Investigate whether there is a significant relationship between reviews and ratings 

The combination with the best performance is already uncommented, feel free to check model performance
on different combination by uncommenting the selected combination and commenting out other combinations

Assuming ratings 1,2,3 = cons; rating 4 = combine, ratings 5 is our current combination
Following is the regular for loop of the list comprehension:
    
rev = []
for i in reviews.index::
    if reviews["rating"][i] == 5:
        rev.append(reviews["pros"][i])
    elif reviews["rating"][i] == 4:
        rev.append(reviews["pros"][i] + ". " + reviews["cons"][i])
    else:
        rev.append(reviews["cons"][i])
            
reviews["review"] = rev

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Ratings 1,2,3 = cons; rating 4 = combine, ratings 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1 = cons; rating 2,3,4 = combine, ratings 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 3 or reviews["rating"][i] == 2
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1 = cons; rating 2 = combine, ratings 3,4,5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 4 or reviews["rating"][i] == 3
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 2
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1,2 = cons; rating 3,4 = combine, ratings 5 = pros (72% accuracy)
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 3
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1 = cons; rating 2,3 = combine, ratings 4,5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 4 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 3 or reviews["rating"][i] == 2
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1,2 = cons; rating 3 = combine, ratings 4,5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 4 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 3
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1,2 = cons; rating 3，4,5 = combine
#reviews["review"] = [reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 4 or reviews["rating"][i] == 3
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1,3 = cons; rating 2, 4 = combine, ratings 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 2 
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 3 = cons; rating 1, 2, 4 = combine, ratings 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 2 or reviews["rating"][i] == 1
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 3 = cons; rating 2, 4 = combine, ratings 1, 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 1
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 2
#                        else reviews["cons"][i] for i in reviews.index]

## Ratings 1 = cons; rating 2, 4 = combine, ratings 3, 5 = pros
#reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 3
#                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 2
#                        else reviews["cons"][i] for i in reviews.index]

## Combine pros and cons reviews
#reviews["review"] = reviews[["pros", "cons"]].apply(lambda x: ". ".join(x), axis = 1)

# Ratings 3 = cons; rating 1, 4 = combine, ratings 2, 5 = pros
reviews["review"] = [reviews["pros"][i] if reviews["rating"][i] == 5 or reviews["rating"][i] == 2
                        else reviews["pros"][i] + ". " + reviews["cons"][i] if reviews["rating"][i] == 4 or reviews["rating"][i] == 1
                        else reviews["cons"][i] for i in reviews.index]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data preprocessing (train test split, remove non-alphabetic characters, remove digit, 
                    remove emoticons, one hot encode labels, tokenization)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
reviews = reviews[["review", "rating"]]
X = reviews["review"]
Y = reviews["rating"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
#y_train_val = y_train.value_counts()
#y_test_val = y_test.value_counts()
#print (y_train_val)
#print (y_test_val)
    
x_train = x_train.apply(lambda x: preProcessing(x)).reset_index(drop=True)
x_test = x_test.apply(lambda x: preProcessing(x)).reset_index(drop=True)

# Array created for resampling purpose (data imbalanced) and one hot encoding
ytrain_arr = np.array(y_train)
ytest_arr = np.array(y_test)

# One hot encode y
y_train = to_categorical(ytrain_arr)
y_test = to_categorical(ytest_arr)

max_features = 20000
maxlen = 200
embed_size = 300
batch_size = 128
epochs = 20
val_split = 0.1

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
tokenized_train = tokenizer.texts_to_sequences(x_train)
tokenized_test = tokenizer.texts_to_sequences(x_test)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Each indexed-reviews has different length, we have to feed data that has a fixed number of features (consistent length)
Find best "maxlen" to set. If we put it too short, we might lose some useful feature that could cost us some accuracy points down the path.
If set it too long, our LSTM cell will have to be larger to store the possible values or states.
Code below plots a histogram showing the distribution of the length of the reviews in our training dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#totalNumWords = [len(review) for review in tokenized_train]
#plt.hist(totalNumWords,bins = np.arange(0,410,10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
#plt.xlabel('Review length')
#plt.ylabel('Frequency')
#plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
As we can see most of the review length is about 30+, but just in case we set the maxlen to 300
Use padding to make shorter reviews as long as the others by filling zeros and trim longer reviews to the maxlen specified. 
Can play with padding and truncating parameters (default padding & truncating = "pre")
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train = pad_sequences(tokenized_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(tokenized_test, maxlen=maxlen, padding='post', truncating='post')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Scatter plot for visualizing our data points in a 2d space
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#plot_2d_space(x_train, ytrain_arr)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Imbalanced data resampling method
Choice on first parameters: under (TomekLinks), over (SMOTE), combined (SMOTETomek)
SMOTETomek = combined under and over sampling; SMOTE = over sampling; TomekLinks = under sampling
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#x_train, y_train = imbalanced_resampling("under", x_train, ytrain_arr)
#unique, counts = np.unique(y_train, return_counts=True)
#print (dict(zip(unique, counts)))

## Uncomment the following line if any resampling method is used
#y_train = to_categorical(y_train)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read the glove word vectors into a dictionary from word->vector.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
embeddings_index = loadEmbedding("glove")
#embeddings_index = loadEmbedding("fasttext")
#embeddings_index = loadEmbedding("word2vec")

    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. 
We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
embedding_matrix = createEmbeddingMatrix(max_features, embed_size, embeddings_index, tokenizer)

            
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Bidirectional LSTM + 1DConvolution with two fully connected layers. 
We add some dropout to the LSTM, after globalMaxPooling and after first dense layer as 2 epochs is enough to overfit.
Comment out CudNNLSTM and uncomment out regular LSTM if GPU is not available
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# shape: A shape tuple (integer), not including the batch size. 
# For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)(inp)
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
x = Dense(y_train.shape[1], activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary() 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks = [early_stopping]);

# summarize history for accuracy and loss
modelHistory(history)
y_pred = model.predict([x_test], batch_size=1024, verbose=1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Testing accuracy
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
acc = sum([np.argmax(y_test[i])==np.argmax(y_pred[i]) for i in range(x_test.shape[0])])/x_test.shape[0]
print ("Model accuracy:", acc)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Plot confusion matrix
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
y_pred_copy = np.copy(y_pred)
for row in range(y_pred_copy.shape[0]):
    y_pred_copy[row] = (y_pred_copy[row] == np.amax(y_pred_copy[row])).astype(int)
    
np.set_printoptions(precision=2)
plot_cf(y_test.argmax(axis=1), y_pred_copy.argmax(axis=1), "LSTM", normalize = True)
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Classification report (precision, recall, f1, micro&macro avg, weighted avg)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Classification Report')
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
