#! /usr/bin/python
import time
start_time = time.time()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd, numpy as np
from math import log
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC

# As of NumPy 1.7, np.dot is not aware of sparse matrices, therefore using it will result on unexpected results or errors. 
# The corresponding dense matrix should be obtained first instead. but then all the performance advantages would be lost. 
# Notice that it returned a matrix, because todense returns a matrix.
# The CSR format is specially suitable for fast matrix vector products.
from scipy.sparse import csr_matrix	

# read in training data
df = pd.read_csv("reviews_tr.csv")
data, labels = np.array(df['text'][0:200000]), np.array(df['label'][0:200000])
# read in test data
df = pd.read_csv("reviews_te.csv")
testdata, testlabels = np.array(df['text'][0:200000]), np.array(df['label'][0:200000])

##################################### data representation #############################################
# unigram tf
vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')  # change to this pattern so that "1 character word" still counts
data_tf = vectorizer.fit_transform(data) # (1000000, 207429)
testdata_tf = vectorizer.transform(testdata)

# tf-idf
transformer = TfidfTransformer(smooth_idf=False, norm=None)
data_tfidf = transformer.fit_transform(data_tf)
# implementation of sklearn when 'smooth_idf=False is: loge(D/x) +1, so need revert it back to as per assignment specification : log10 (D/x) (x is the #documents contains the word)
data_tfidf = data_tfidf - data_tf	# eliminate (+1)
data_tfidf = data_tfidf / log(10) # change log base from e to 10
testdata_tfidf = transformer.transform(testdata_tf)
testddta_tfidf = testdata_tfidf - testdata_tf
testdata_tfidf = testdata_tfidf / log(10)

# bigram tf representation
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern='(?u)\\b\\w+\\b', min_df=1)
data_tf_bigram = bigram_vectorizer.fit_transform(data)
testdata_tf_bigram = bigram_vectorizer.transform(testdata)

# # my own choosing of data representation: trigram
# trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), token_pattern='(?u)\\b\\w+\\b', min_df=1)
# data_tf_trigram = trigram_vectorizer.fit_transform(data)
# testdata_tf_trigram = trigram_vectorizer.transform(testdata)

# my own choosing of data representation: tfidf with natural log, normalization and smooth, and ignore "one char" word
transformer2 = TfidfTransformer(smooth_idf=True, norm=u'l2')
data_tfidf2 = transformer2.fit_transform(data_tf)
testdata_tfidf2 = transformer2.transform(testdata_tf)

########################## Average Perceptron #######################################
def avg_perceptron(data, labels):
	weight, threshold = np.zeros( data[0, :].shape[1]), np.float64(0)

	# first pass
	data, labels = shuffle(data, labels)
	for i in tqdm(xrange(len(labels))):
		if ( data[i, :].dot(weight) + threshold ) * labels[i] <= 0:
			weight += labels[i] * data[i, :] # current type is <class 'numpy.matrixlib.defmatrix.matrix'>, 
			weight = np.array(weight).flatten() # so need to convert it to vector by .flatten()
			threshold += labels[i]

	# start do average on the second pass
	final_weight, final_threshold = weight, threshold

	# second pass, calculate the average classifier
	data, labels = shuffle(data, labels)
	for i in tqdm(xrange(len(labels))):
		if (data[i, :].dot(weight) + threshold) * labels[i] <= 0:
			weight += labels[i] * data[i, :]
			weight = np.array(weight).flatten()
			threshold += labels[i]
		final_weight += weight
		final_threshold += threshold
	final_weight, final_threshold = final_weight / np.float64(len(labels)+1), final_threshold / np.float64(len(labels)+1)
	return final_weight, final_threshold
#######################################################################################

################ Naive Bayes ##################
def naive_bayes(data, labels):
	data1 = data
	data1[data1!=0]=1
	clf = MultinomialNB()
	# train the model
	clf.fit(data1, labels)
	return clf
###############################################

############# My own method choosing: LinearSVC with unigram data representation ###############
def mySVC(data, labels):
	clf = svm.LinearSVC()
	clf.fit(data, labels)
	return clf
#################################################################################################


################################################ Cross Validation 5-fold ############################################################
linearSVC_err, naive_bayes_err, avg_perceptron_tf, avg_perceptron_tfidf, avg_perceptron_bigram, avg_perceptron_tfidf2 = 0, 0, 0, 0, 0, 0
kf = KFold(n_splits=5)
for train_idx, test_idx in kf.split(data_tf):

	# LinearSVC: This is my own method of choosing: LinearSVC with unigram data representation
	# avg [0.1133, 0.113, 0.112775, 0.1135, 0.1102] = 0.112555, time = 50.8332090378s
	clf = mySVC(data_tfidf2[train_idx], labels[train_idx])
	linearSVC_err += np.float64(clf.predict(data_tfidf2[test_idx]) != labels[test_idx] ).sum() / len(labels[test_idx])


	# This is the sklearn naive bayes (multinomial) with unigram representation
	# avg [0.140575, 0.139775, 0.1443, 0.1417, 0.13855] = 0.14098 , time = 45.16s
	clf = naive_bayes(data_tf[train_idx], labels[train_idx])
	data_tf1 = data_tf
	data_tf1 [data_tf1!=0] = 1
	naive_bayes_err += np.float64((clf.predict(data_tf[test_idx]) != labels[test_idx]).sum()) / len(labels[test_idx])

	# Below are the average perceptron with four different data representation
	labels[labels==0] = -1

	# avg_perceptron with unigram representation
	# results: avg [0.124325. 0.11862, 0.1214, 0.1118, 0.129175 ] = 0.12005, time = 314s!
	weight, threshold = avg_perceptron(data_tf[train_idx], labels[train_idx])
	avg_perceptron_tf += np.float64((( data_tf[test_idx].dot(weight) + threshold) * labels[test_idx] <= 0 ).sum())/ len(labels[test_idx])

	# avg_perceptron with tfidf representation
	# results: avg[0.145575, 0.14725, 0.12875, 0.136475, 0.1474] = 0.1411, time = 277.23853302s!
	weight, threshold = avg_perceptron(data_tfidf[train_idx], labels[train_idx])
	avg_perceptron_tfidf += np.float64((( data_tfidf[test_idx].dot(weight) + threshold) * labels[test_idx] <= 0 ).sum())/ len(labels[test_idx])

	# try avg_perceptron with bigram representation
	# results: avg[0.155, 0.146125, 0.112725, 0.14465, 0.118] = 0.1353, time = 6234.6195631s!
	weight, threshold = avg_perceptron(data_tf_bigram[train_idx], labels[train_idx])
	avg_perceptron_bigram += np.float64((( data_tf_bigram[test_idx].dot(weight) + threshold) * labels[test_idx] <= 0 ).sum())/ len(labels[test_idx])

	# # try avg_perceptron with trigram representation
	# # results: avg[0.16665, 0.1567, 0.158225, 0.147175, 0.1573] = 0.1572, time = 55026.1957531s!
	# weight, threshold = avg_perceptron(data_tf_trigram[train_idx], labels[train_idx])
	# avg_perceptron_trigram += np.float64((( data_tf_trigram[test_idx].dot(weight) + threshold) * labels[test_idx] <= 0 ).sum())/ len(labels[test_idx])

	# try avg_perceptron with my own variant tfidf
	# avg[0.113, 0.11915, 0.124175, 0.12435, 0.1279] = 0.1217, time = 324.522380114s
	weight, threshold = avg_perceptron(data_tfidf2[train_idx], labels[train_idx])
	avg_perceptron_tfidf2 += np.float64((( data_tfidf2[test_idx].dot(weight) + threshold) * labels[test_idx] <= 0 ).sum())/ len(labels[test_idx])

	labels[labels==-1] = 0

linearSVC_err, naive_bayes_err, avg_perceptron_tf, avg_perceptron_tfidf, avg_perceptron_bigram, avg_perceptron_tfidf2 = linearSVC_err/5, naive_bayes_err/5, avg_perceptron_tf/5, avg_perceptron_tfidf/5, avg_perceptron_bigram/5, avg_perceptron_tfidf2/5
print "linearSVC_err", linearSVC_err, "naive_bayes_err", naive_bayes_err, "avg_perceptron_tf", avg_perceptron_tf, "avg_perceptron_tfidf", avg_perceptron_tfidf, "avg_perceptron_bigram", avg_perceptron_bigram, "avg_perceptron_tfidf2", avg_perceptron_tfidf2


# The best method I finally choose to train the classifier is the "LinearSVC with my own variant tf-idf data representation"
# final training error rate: 0.09589, final test error rate: 0.12065, time = 110.663964033s
labels[labels==0], testlabels[testlabels==0] =-1, -1
clf = mySVC(data_tfidf2, labels)
train_err, test_err = np.float64(clf.predict(data_tfidf2) != labels).sum() / len(labels), np.float64(clf.predict(testdata_tfidf2) != testlabels ).sum() / len(testlabels)
labels[labels==0], testlabels[testlabels==-1] = 0, 0
print "train=", train_err, "test=", test_err

print "Program runs", time.time() - start_time, "s!"
