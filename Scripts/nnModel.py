import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Conv1D, Conv2D, MaxPooling1D, GRU, Embedding, Dropout, Concatenate, Reshape
from keras.layers import Flatten, Bidirectional, GlobalMaxPool1D, GlobalMaxPool2D, SpatialDropout1D
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
This model takes the 5 aspects rating as input and used them to predict the overall ratings
Comment out CudNNLSTM and uncomment out regular LSTM if gpu is not available
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# shape: A shape tuple (integer), not including the batch size. eg. shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
# In our case, input tensor for sequences of 1 timestep, each containing a 5-dimensional vector
maxlen = 5
batch_size = 128
epochs = 20
val_split = 0.1

inp = Input(shape=(1, maxlen))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inp)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(y_train_rate.shape[-1], activation="softmax")(x)
model_rate = Model(inputs=inp, outputs=x)
model_rate.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model_rate.summary() 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model_rate.fit(x_train_rate, y_train_rate, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks = [early_stopping]);
#rating_pred_test = model.predict([x_test_rate], batch_size=1024, verbose=1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Save model(aspects rating) to JSON
GPU available: uncoment line 104 - 110
GPU unavailable: uncoment line 113 - 119
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# serialize model to JSON (GPU version)
model_json = model_rate.to_json()
with open("ratings_model_gpu.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model_rate.save_weights("ratings_model_gpu.h5")

## serialize model to JSON (CPU version)
#model_json = model_rate.to_json()
#with open("ratings_model_cpu.json", "w") as json_file:
#    json_file.write(model_json)
#    
## serialize weights to HDF5
#model_rate.save_weights("ratings_model_cpu.h5")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Extract pros and cons reviews and combine them together as one review
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
Data preprocessing (remove, emoticons, remove non-alphabetic characters, remove digit, 
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

# Save tokenizer object
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
tokenized_train = tokenizer.texts_to_sequences(x_train_rev)
tokenized_test = tokenizer.texts_to_sequences(x_test_rev)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Use padding to make shorter x_train_rev as long as the others by filling zeros.
Use padding to trim longer x_train_rev to the maxlen specified. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
x_train_rev = pad_sequences(tokenized_train, maxlen=maxlen, padding='post', truncating='post') 
x_test_rev = pad_sequences(tokenized_test, maxlen=maxlen, padding='post', truncating='post')


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Read pre-trained word vectors into a dictionary from word->vector.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
embeddings_index = loadEmbedding("glove")
#embeddings_index = loadEmbedding("fasttext")
#embeddings_index = loadEmbedding("word2vec")
    

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
Use these vectors to create our embedding matrix, with random initialization for words that aren't in word vector. 
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

# Single channel architecture LSTM-CNN
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

## Multichannel architecture LSTM-CNN
#y = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)(inp)
#z = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
#y = SpatialDropout1D(0.2)(y)
#z = SpatialDropout1D(0.2)(z)
#b1 = Bidirectional(CuDNNLSTM(50, return_sequences=True))(y)
#b2 = Bidirectional(CuDNNLSTM(50, return_sequences=True))(z)
##b1 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(y)
##b2 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(z)
#x = Concatenate()([b1, b2])
#x = Reshape((maxlen, 100, 2))(x)
#conv1 = Conv2D(200, kernel_size = (3, 100), activation = 'relu')(x)
#conv2 = Conv2D(200, kernel_size = (4, 100), activation = 'relu')(x)
#conv3 = Conv2D(200, kernel_size = (5, 100), activation = 'relu')(x)
#conv1 = GlobalMaxPool2D()(conv1)
#conv2 = GlobalMaxPool2D()(conv2)
#conv3 = GlobalMaxPool2D()(conv3)

x = Concatenate()([conv1,conv2,conv3])
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(y_train_rev.shape[1], activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary() 
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(x_train_rev, y_train_rev, batch_size=batch_size, epochs=epochs, validation_split= val_split, callbacks = [early_stopping]);

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

    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Save model to JSON
GPU available: uncoment line 285 - 290
GPU unavailable: uncoment line 292 - 297
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# serialize model to JSON (GPU version)
model_json = model.to_json()
with open("review_model_gpu.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("review_model_gpu.h5")

## serialize model to JSON (CPU version)
#model_json = model.to_json()
#with open("review_model_cpu.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("review_model_cpu.h5")