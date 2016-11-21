#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""


import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import f1_score
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

s0 = False
all_file = True
top_k = False
emb_file = "../emb/blogcatalog/blogcatalog_cfi.emb"
mat_file = "../graph/blogcatalog.mat"
emb = None
startfrom0=True
topk=False

def usage():
    print '''
        -f: input embedding file
        -m: input mat file
        -s: index start from 0 or 1
        -t: top-k method or not
        -h: help function
        -a: check scoring for all files in emb directory
        '''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hf:sm:ta", ["file="])
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
    if opt in ("-a"):
        all_file = True
    elif opt in ("-h"):
        usage()
        sys.exit()

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}


def label2onehot(x, classes):
    out = []
    for index, i in enumerate(x):
       out.append(numpy.sum(numpy.eye(classes)[i], axis=0))
    return numpy.array(out)

def score(emb, startfrom0=False, topk=False):

    # 0. Files
    #embeddings_file = "blogcatalog.embeddings"
    list_of_files = glob.glob('../emb/kaggle/*.emb')
    matfile = mat_file
    embeddings_file = emb_file

    # 2. Load labels
    mat = loadmat(matfile)
    A = mat['network']
    graph = sparse2graph(A)
    labels_matrix = mat['group']

    if startfrom0:
        index_align = 0
    else:
        index_align = 1

    features_matrix_array = []
    dw_features_matrix_array = {}
    cf_features_matrix_array = {}
    cfi_features_matrix_array = {}

    if all_file:
        for f in list_of_files:
            embed = numpy.loadtxt(f, skiprows=1)
            features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node+index_align), 1:][0,0] for node in range(len(graph))])
            features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])
            if os.path.basename(os.path.splitext(f)[0]).split('_')[-1] == 'cfi':
                cfi_features_matrix_array['cfi']=features_matrix
            elif os.path.basename(os.path.splitext(f)[0]).split('_')[-1] == 'cf':
                cf_features_matrix_array['cf']=features_matrix
            else:
                nw=int(os.path.basename(os.path.splitext(f)[0]).split('_')[-1])
                dw_features_matrix_array[nw]=features_matrix
        features_matrix_array.append(dw_features_matrix_array)
        features_matrix_array.append(cf_features_matrix_array)
        features_matrix_array.append(cfi_features_matrix_array)

    else:
        if emb is None:
            # 1. Load Embeddings
            embed = numpy.loadtxt(embeddings_file, skiprows=1)
            features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node+index_align), 1:][0,0] for node in range(len(graph))])
            features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])
        else:
            features_matrix = emb
        features_matrix_array.append(features_matrix)

    res = []

    training_percents = [0.3, 0.5, 0.9]
    # uncomment for all training percents
    #training_percents = numpy.asarray(range(1,10))*.1
    for emb in features_matrix_array:
        score_array = {}
        for key in emb.keys():
            emb_buf = emb[key]
            # 3. to score each train/test group
            all_results = defaultdict(list)

            # 2. Shuffle, to create train/test groups
            shuffles = []
            number_shuffles = 2
            for x in range(number_shuffles):
                shuffles.append(skshuffle(emb_buf, labels_matrix))

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
                        clf = TopKRanker(LogisticRegression(max_iter=500,))
                    else:
                        clf = OneVsRestClassifier(LogisticRegression(max_iter=500))

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


            print 'Results, using embeddings of dimensionality', X.shape[1]
            print '-------------------'
            for train_percent in sorted(all_results.keys()):
                print 'Train percent:', train_percent
                for x in all_results[train_percent]:
                    print  x
                print '-------------------'

            score_array[key]=all_results
        res.append(score_array)

    dw_res, cf_res, cfi_res=res[0], res[1], res[2]

    averages = ["micro", "macro", "samples", "weighted"]
    percent = [0.3, 0.5, 0.9]
    for average in averages:
        for p in percent:
            plt.figure()
            y_value_dw = [dw_res[k][p][0][average] for k in sorted(dw_res.keys())]
            y_value_cf = [cf_res['cf'][p][0][average] for k in sorted(dw_res.keys())]
            y_value_cfi = [cfi_res['cfi'][p][0][average] for k in sorted(dw_res.keys())]
            plt.plot(y_value_dw, 'bo-')
            plt.plot(y_value_cf, 'ro-')
            plt.plot(y_value_cfi, 'go-')
            plt.grid(True)
            plt.xlabel('number of walks at 10, 20, 50, 100')
            plt.ylabel('score')
            plt.title('percentage: %f, metric: %s' %(p, average) )
            plt.savefig("p%.1f_%s.png" % (p,average))
            #plt.show()





score(emb, startfrom0=s0, topk=top_k)
