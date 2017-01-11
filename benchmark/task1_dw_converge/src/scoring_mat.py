#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""


import numpy
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import accuracy_score
import scipy
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

import glob
from collections import defaultdict

import matplotlib.pyplot as plt

import sys
import getopt
import os, struct
from os.path import basename
import fnmatch

from utils import TopKRanker, sparse2graph, label2onehot

s0 = False
top_k = False
random_enable = False
emb_file = "../data/blogcatalog_20.emb"
mat_file = "../data/blogcatalog.mat"
out_file = "../result/blogcatalog"
emb = None
startfrom0=False
topk=False
classifier='log'
k=10

def usage():
    print '''
        -f: input embedding file
        -m: input mat file
        -s: index start from 0 or 1
        -t: top-k method or not
        -h: help function
        -c: classifier: svm/log
        -k: k-fold cross validation
        -o: output file
        '''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:sm:tk:c:o:", ["file="])
except getopt.GetoptError, err:
    print 'invalid command line'
    usage()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-f"):
        emb_file = arg
    if opt in ("-m"):
        mat_file = arg
    if opt in ("-s"):
        s0 = True
    if opt in ("-t"):
        top_k = True
    if opt in ("-k"):
        k = int(arg)
    if opt in ("-c"):
        classifier = arg
    if opt in ("-o"):
        out_file = arg
    elif opt in ("-h"):
        usage()
        sys.exit()


def score(emb, startfrom0=False, topk=False, k=2, classifier=classifier):

    # 0. Files
    #embeddings_file = "blogcatalog.embeddings"
    matfile = mat_file
    embeddings_file = emb_file
    output_file = out_file

    # 2. Load labels
    mat = loadmat(matfile)
    A = mat['network']
    graph = sparse2graph(A)
    labels_matrix = mat['label']

    if startfrom0:
        index_align = 0
    else:
        index_align = 1

    if emb is None:
        # 1. Load Embeddings
        embed = numpy.loadtxt(embeddings_file, skiprows=1)
        features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node+index_align), 1:][0,0] for node in range(A.shape[0])])
        features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])
    else:
        features_matrix = emb

    shuffles = []
    number_shuffles = k

    for x in range(number_shuffles):
      shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group

    results = []
    training_percents = [1, 2, 3, 4]
    # uncomment for all training percents
    #training_percents = numpy.asarray(range(1,10))*.1
    print 'Averaged crossvalidation results, using embeddings of dimensionality', features_matrix.shape[1]
    for samples in training_percents:
        acc = 0
        for shuf in shuffles:

            X, y = shuf

            training_size = int(samples*y.shape[1])
            label = y
            y = numpy.argmax(y, axis=1)

            if random_enable:
                X_train = X[:training_size, :]
                y_train = y[:training_size]
                X_test = X[training_size:, :]
                y_test = y[training_size:]
            else:
                xs = []
                xts = []
                for i in range (label.shape[1]):
                    y_ = label[:,i]
                    x_ = numpy.array(range(y_.shape[0]))
                    class_xs = []
                    min_elems = None

                    for yi in numpy.unique(y_):
                        elems = x_[(y_==yi)]
                        class_xs.append((yi, elems))
                        if min_elems == None or elems.shape[0] < min_elems:
                            min_elems = elems.shape[0]

                    xs = numpy.concatenate((xs, elems[:samples]))
                    xts = numpy.concatenate((xts, elems[samples:]))
                idx_train = xs.astype(int)
                idx_test = xts.astype(int)
                X_train = X[idx_train]
                y_train = y[idx_train]
                X_test = X[idx_test]
                y_test = y[idx_test]

            if classifier == 'log':
                clf = LogisticRegression(max_iter=500,)
            elif classifier == 'svm':
                clf = LinearSVC()

            if sum(y_train) == 0 or sum(y_train) == training_size:
                acc += 0.5
            else:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc += accuracy_score(preds, y_test)

        avg_acc = acc/k
        results.append(avg_acc)
        print '-------------------', ' number samples:', samples, 'accuracy:', avg_acc, '----------------'

    filename = os.path.splitext(output_file)[0]
    filename = '%s_result.mat' % filename
    scipy.io.savemat(filename, mdict={'average': results})


score(emb, startfrom0=s0, topk=top_k, k=k)
