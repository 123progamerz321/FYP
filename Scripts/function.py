# -*- coding: utf-8 -*-
"""
Created on Wed May 22 02:37:22 2019

@author: A
"""
import re
import matplotlib.pyplot as plt
import numpy as np
import gensim.models.keyedvectors as word2vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

def preProcessing(text, stopword = False, stem = False):
    """
    This function removes non-alphabetic characters, digits and emoticons in the review, and converts all characters to lowercase
    Optionally, remove stop words and shorten words to their stems (not recommended, performance measures will drop)
    """
    
    specialCharRemoval = re.compile(r'[^a-z\d ]', re.IGNORECASE)
    replaceDigit = re.compile(r'\d+', re.IGNORECASE)
    text = specialCharRemoval.sub('', text)
    text = replaceDigit.sub('', text)
    text = " ".join(text.lower().split())
    
    # Optionally, remove stop words
    if stopword:
        stops = set(stopwords.words("english"))
        stops.remove("not")
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    return text


def plot_2d_space(X, y, label='Classes'):   
    """
    Scatter plot for visualizing our data points in a 2d space
    """
    
    colors = ['#1F77B4', '#FF7F0E', 'c', 'm', 'y']
    markers = ['o', 's', 'v', 'p', 'h']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
    
def imbalanced_resampling(method, x, y):
    if method == "under":
        sampling = TomekLinks(sampling_strategy = "auto")
    elif method == "over":
        sampling = SMOTE(ratio='auto')
    elif method == "combined":
        sampling = SMOTETomek()
    else:
        return x, y
    
    X, Y =  sampling.fit_sample(x, y)
    return X, Y


def loadEmbedding(filename):
    """
    This function loads different embedding based on user's choice
    Including glove, fasttext and word2vec
    """
    
    if filename == "glove":
        embeddingFile = "C:/Users/A/Desktop/FYP/Embeddings/glove.840B.300d.txt"
    elif filename == "fasttext":
        embeddingFile = "C:/Users/A/Desktop/FYP/Embeddings/crawl-300d-2M.vec"
    elif filename == "word2vec":
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format("C:/Users/A/Desktop/FYP/Embeddings/GoogleNews-vectors-negative300.bin", binary=True)
       
    embeddings_index = {}
    if filename == "glove" or filename == "fasttext":
        with open(embeddingFile, encoding = "utf-8") as f:
            for line in f:
                values = line.rstrip().rsplit(" ")
                word = values[0]
                # dtype : data-type, optional. By default, the data-type is inferred from the input data.
                coefs = np.asarray(values[1:], dtype = "float32")
                embeddings_index[word] = coefs
        
    else:
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)
            
    print ("Total word vectors = ", len(embeddings_index))            
    return embeddings_index


def createEmbeddingMatrix(max_features, embed_size, embeddings_index, tokenizer):
    """
    This functions creates our embedding matrix, with random initialization for words that aren't in GloVe. 
    We'll use the same mean and stdev of embeddings the GloVe/Fasttext/Word2Vec has when generating the random init.
    """
    embeddedCount = 0
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        i -= 1
        if i < max_features:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount += 1
                
    print ("Total embedded:", embeddedCount, "common words")
    return embedding_matrix
                   
def modelHistory(history):
    """
    This function summarises history for accuracy and loss
    """
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def plot_cf(y_test, y_pred, classifier, normalize = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    matrix = confusion_matrix(y_test, y_pred)
    classes = unique_labels(y_test, y_pred)
    
    title = "Confusion matrix, without normalization"
    if normalize:
        title = "Normalized confusion matrix - " + classifier
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()  
    return ax