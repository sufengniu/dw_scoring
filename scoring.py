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

from collections import defaultdict

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


# 0. Files
#embeddings_file = "blogcatalog.embeddings"
matfile = "blogcatalog.mat"
embeddings_file = "blogcatalog.embeddings_64.embeddings"

# 1. Load Embeddings
#model = Word2Vec.load_word2vec_format(embeddings_file, binary=False,
#                                      norm_only=False)
embed = numpy.loadtxt(embeddings_file, skiprows=1)

# 2. Load labels
mat = loadmat(matfile)
A = mat['network']
graph = sparse2graph(A)
labels_matrix = mat['group']


# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
#features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])
#features_matrix = embed[:,1:]

features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node), 1:][0,0] for node in range(len(graph))])
features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])

# 2. Shuffle, to create train/test groups
shuffles = []
number_shuffles = 2
for x in range(number_shuffles):
  shuffles.append(skshuffle(features_matrix, labels_matrix))

# 3. to score each train/test group
all_results = defaultdict(list)

training_percents = [0.3, 0.5, 0.9]
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
    y_train_onehot = label2onehot(y_train, 39)

    #assert sum(len(l) for l in y_train) == y_train_.nnz

    X_test = X[training_size:, :]
    y_test_ = y[training_size:]

    y_test = [[] for x in xrange(y_test_.shape[0])]

    cy =  y_test_.tocoo()
    for i, j in izip(cy.row, cy.col):
        y_test[i].append(j)

    #y_test_onehot = mlb.fit_transform(y_test)
    y_test_onehot = label2onehot(y_test, 39)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=500))
    clf.fit(X_train, y_train_onehot)

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
