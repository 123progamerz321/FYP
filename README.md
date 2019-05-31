# FYP-Review Rating Prediction and Analysis

## Steps to run the program

Step 1: Download as ZIP, extract it to the Desktop

Step 2: Rename the folder to FYP (remove the -master suffix)

Step 3: Download and Install Anaconda https://www.anaconda.com/distribution/

Step 4: Download Glove embeddings http://nlp.stanford.edu/data/glove.840B.300d.zip, and extract it to ..\Desktop\FYP\Embeddings

Step 5: Download FastText https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m, and extract it to ..\Desktop\FYP\Embeddings

Step 6: Download Word2Vec https://www.kaggle.com/sandreds/googlenewsvectorsnegative300#GoogleNews-vectors-negative300.bin, and extract it to ..\Desktop\FYP\Embeddings

Step 7: Open Anaconda Navigator, go to Environments and Install the following packages:
- pandas
- matplotlib
- numpy
- gensim
- nltk
- scikit-learn
- keras (if GPU is not available)
- keras-gpu
- tensorflow
- tensorflow-gpu (if GPU is not available)
- imblearn (not provided in Environments' list of packages, instead open Anaconda Prompt and type "conda install -c conda-forge imbalanced-learn") 
- h5py

Step 8: Open Anaconda Prompt, Navigate the path to ..\Desktop\FYP\Scripts. For example in our device:
```
C:\Users\user\Desktop\FYP\Scripts
```

Step 9: Enter the following command:
```
python main.py
```

## For data exploratory/experimentation/helper purposes
- function.py (Contains all helper functions)
- nnModel.py (Neural network training)
- svmModel.py (SVM training)
- unitTesting.py
- testCombined.py (Data experimentation)
- revRatingCorrelation.py (Data experimentation)
- splitTestOnPredRate.py (Data experimentation)

## Saved Models, Model's weights, Tokenizer, TFIDF Vectorizer(GPU&CPU)
- all files with json, pickle, h5 and sav extension
