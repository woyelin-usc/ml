#!/usr/local/Cellar/python/2.7.12/bin/python
import time
start_time = time.time()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree

# read the .csv file into the data frame 'df'
df = pd.read_csv('reviews_tr.csv')
data, labels = df['text'], df['label']
df = pd.read_csv('reviews_te.csv')
testdata, testlabels = df['text'], df['label']

# construct tf feature matrices for data and testdata
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # change to this pattern so that "1 character word" counts
data = vectorizer.fit_transform(data)
testdata = vectorizer.transform(testdata)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, labels)
print (clf.predict(testdata) != testlabels) / len(testlabels)


print time.time() - start_time,  "s"