#!/usr/local/Cellar/python/2.7.12/bin/python
import time
start_time = time.time()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, numpy as np
from sklearn.naive_bayes import MultinomialNB

# read the .csv file into the data frame 'df'
df = pd.read_csv('mytrain.csv')

# explicilty convert the reviews into a numpy 
data, labels = np.array(df['text']), np.array(df['label'])

# construct tf feature matrices
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # change to this pattern so that "1 character word" counts
data = vectorizer.fit_transform(data)

print vectorizer.vocabulary_
print time.time() - start_time, "second"











# # read in the test file
# df = pd.read_csv('reviews_te.csv')
# testdata, testlabels = np.array(df['text']), np.array(df['label'])
# # here is to extract features from test data use the training model, so use 'transform' not 'fit_transform'
# testdata = vectorizer.fit_transform(testdata)


# # test naive-bayes on unigram tf
# clf = MultinomialNB()
# clf.fit(data, labels)
# print clf.predict(testdata)
# #print (clf.predict(testdata) != testlabels).sum() / len(testlabels)












# # # Converse mapping from feature name to column index (which is the column idex of feature 'love'?)
# # print vectorizer.vocabulary_.get('love');

# # construct tf-idf feature matrices
# transformer = TfidfTransformer(smooth_idf=False)
# tfidf = transformer.fit_transform(tf)

# # construct the bigram feature matrices 
# vectorizer = CountVectorizer(ngram_range=(2,2))
# tf_bigram = vectorizer.fit_transform(texts)

# # I choose to construct the trigram feature matrices
# vectorizer = CountVectorizer(ngram_range=(3,3))
# tf_trigram = vectorizer.fit_transform(texts)

