
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
from sklearn.metrics import accuracy_score

import glob
from collections import defaultdict

import matplotlib.pyplot as plt

import sys
import getopt
import os, struct
from os.path import basename
import fnmatch

from utils import TopKRanker, sparse2graph, label2onehot



embeddings_file = 'blog.emb'

mat = loadmat('blogs_n.mat')
A = mat['network']
graph = sparse2graph(A)
labels_matrix = mat['labels']

embed = numpy.loadtxt(embeddings_file, skiprows=1)
features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node), 1:][0,0] for node in range(A.shape[0])])
features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])

labels_matrix = numpy.reshape(labels_matrix, (-1))

clf = LogisticRegression(max_iter=200,)

y_train = labels_matrix[[1,-1]]
x_train = features_matrix[[1,-1]]
clf.fit(x_train, y_train)
y_test=clf.predict(features_matrix)
print accuracy_score(labels_matrix, y_test)

