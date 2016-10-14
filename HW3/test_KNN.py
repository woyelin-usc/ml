#!/usr/local/Cellar/python/2.7.12/bin/python
import time
start_time = time.time()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, numpy as np
from sklearn.neighbors import KNeighborsClassifier

# read the .csv file into the data frame 'df'
df = pd.read_csv('mytrain.csv')
data, labels = df['text'], df['label']
df = pd.read_csv('mytest.csv')
testdata, testlabels = df['text'], df['label']

# construct tf feature matrices for data and testdata
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # change to this pattern so that "1 character word" counts
data = vectorizer.fit_transform(data)
testdata = vectorizer.transform(testdata)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data, labels)
print (neigh.predict(testdata) != testlabels).sum() / len(testlabels)