#!/usr/local/Cellar/python/2.7.12/bin/python
import time
start_time = time.time()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, numpy as np
from math import log

corpus = [ 'This is my dog', 'A dog is a dog']

# construct tf feature matrices
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # change to this pattern so that "1 character word" counts
data = vectorizer.fit_transform(corpus)
print data.toarray()

# tf-idf
transformer = TfidfTransformer(smooth_idf=False, norm=None)
tfidf = transformer.fit_transform(data)
# implementation of sklearn when 'smooth_idf=False is: loge(D/x) +1, so need revert it back to : log10 (D/x) (x is the #documents contains the word)
tfidf = tfidf - data	# eliminate (+1)
tfidf = tfidf / log(10) # change log base from e to 10
print tfidf.toarray()

# test bigram feature matrices
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern='(?u)\\b\\w+\\b', min_df=1)
# analyze = bigram_vectorizer.build_analyzer()
tf_bigram = bigram_vectorizer.fit_transform(corpus)

# test trigram feature matrices
trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), token_pattern='(?u)\\b\\w+\\b', min_df=1)
tf_trigram = trigram_vectorizer.fit_transform(corpus)
