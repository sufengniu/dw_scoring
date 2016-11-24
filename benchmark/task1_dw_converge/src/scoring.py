#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""


import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import f1_score
import scipy
from scipy.io import loadmat
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
emb_file = "../data/blogcatalog.emb"
mat_file = "../data/blogcatalog.mat"
out_file = "../result/blogcatalog"
emb = None
startfrom0=True
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
        k = int(k)
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
    labels_matrix = mat['group']

    if startfrom0:
        index_align = 0
    else:
        index_align = 1

    if emb is None:
        # 1. Load Embeddings
        embed = numpy.loadtxt(embeddings_file, skiprows=1)
        features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node+index_align), 1:][0,0] for node in range(len(graph))])
        features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])
    else:
        features_matrix = emb

    shuffles = []
    number_shuffles = k
    for x in range(number_shuffles):
      shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    all_results_m = []
    training_percents = [0.1, 0.5, 0.9]
    # uncomment for all training percents
    #training_percents = numpy.asarray(range(1,10))*.1
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in xrange(y_train_.shape[0])]

            cy =  y_train_.tocoo()
            for i, j in izip(cy.row, cy.col):
                y_train[i].append(j)

            #mlb = MultiLabelBinarizer()
            #y_train_onehot = mlb.fit_transform(y_train)
            y_train_onehot = label2onehot(y_train, labels_matrix.toarray().shape[1])

            #assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for x in xrange(y_test_.shape[0])]

            cy =  y_test_.tocoo()
            for i, j in izip(cy.row, cy.col):
                y_test[i].append(j)

            #y_test_onehot = mlb.fit_transform(y_test)
            y_test_onehot = label2onehot(y_test, labels_matrix.toarray().shape[1])

            if topk:
                if classifier == 'log':
                    clf = TopKRanker(LogisticRegression(max_iter=500,))
                elif classifier == 'svm':
                    clf = TopKRanker(LinearSVC())
            else:
                if classifier == 'log':
                    clf = OneVsRestClassifier(LogisticRegression(max_iter=500,))
                elif classifier == 'svm':
                    clf = OneVsRestClassifier(LinearSVC())

            clf.fit(X_train, y_train_onehot)

            if topk:
                # find out how many labels should be predicted
                top_k_list = [len(l) for l in y_test]
                preds = clf.predict(X_test, top_k_list)
                preds = label2onehot(preds, labels_matrix.toarray().shape[1])
            else:
                preds = clf.predict(X_test)

            results = {}
            averages = ["micro", "macro", "samples", "weighted"]
            for average in averages:
                results[average] = f1_score(y_test_onehot,  preds, average=average)

            all_results[train_percent].append(results)
            all_results_m.append(results)

    m_buf = []
    v_buf = []
    mean_results = defaultdict(list)
    for train_percent in sorted(all_results.keys()):
        res_tmp = {}
        m_tmp = []
        v_tmp = []
        m_tmp.append(train_percent)
        for average in averages:
            res_tmp[average] = numpy.average([all_results[train_percent][j][average] for j in range(k)])
            m_tmp.append(numpy.average([all_results[train_percent][j][average] for j in range(k)]))
            v_tmp.append(numpy.var([all_results[train_percent][j][average] for j in range(k)]))
        mean_results[train_percent].append(res_tmp)
        m_buf.append(m_tmp)
        v_buf.append(v_tmp)

    #if isinstance(os.path.basename(os.path.splitext(embeddings_file)[0]).split('_')[-1], int):
    #    filename = '../result/results_%d.mat' % int(os.path.basename(os.path.splitext(embeddings_file)[0]).split('_')[-1])
    #else:
    #    filename = '../result/results.mat'
    filename = os.path.splitext(output_file)[0]
    filename = '%s_result.mat' % filename
    scipy.io.savemat(filename, mdict={'average': m_buf, 'variance': v_buf, 'origin': all_results_m})

    print 'Averaged crossvalidation results, using embeddings of dimensionality', X.shape[1]
    print '-------------------'
    for train_percent in sorted(mean_results.keys()):
        print 'Train percent:', train_percent
        for x in mean_results[train_percent]:
            print  x
        print '-------------------'


score(emb, startfrom0=s0, topk=top_k, k=k)
