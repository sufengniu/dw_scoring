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

import matplotlib.pyplot as plt

import sys
import getopt
import os, struct
from os.path import basename
import fnmatch


percent_d = [0.1, 0.5, 0.9]

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


def plot_dw(res, percentage=None):
    dw_res=res[0]
    averages = ["micro", "macro", "samples", "weighted"]
    if percentage is None:
        percent = percent_d
    else:
        percent = percentage
    for average in averages:
        for p in percent:
            plt.figure()
            y_value_dw = [dw_res[k][p][average] for k in sorted(dw_res.keys())]
            
            plt.plot(y_value_dw, 'bo-')
            plt.grid(True)
            plt.xlabel('number of walks at 10, 20, 50, 100')
            plt.ylabel('score')
            plt.title('percentage: %f, metric: %s' %(p, average) )
            plt.savefig("p%.1f_%s.png" % (p,average))
            #plt.show()


def plot_res(res, percentage=None):
    dw_res, cf_res, cfi_res, attr_res=res[0], res[1], res[2], res[3]
    averages = ["micro", "macro", "samples", "weighted"]
    if percentage is None:
        percent = percent_d
    else:
        percent = percentage
    for average in averages:
        for p in percent:
            plt.figure()
            y_value_dw = [dw_res[k][p][0][average] for k in sorted(dw_res.keys())]
            y_value_cf = [cf_res['cf'][p][0][average] for k in sorted(dw_res.keys())]
            y_value_cfi = [cfi_res['cfi'][p][0][average] for k in sorted(dw_res.keys())]
            y_value_attr = [attr_res['attr'][p][0][average] for k in sorted(dw_res.keys())]

            plt.plot(y_value_dw, 'bo-')
            plt.plot(y_value_cf, 'ro-')
            plt.plot(y_value_cfi, 'go-')
            plt.grid(True)
            plt.xlabel('number of walks at 10, 20, 50, 100')
            plt.ylabel('score')
            plt.title('percentage: %f, metric: %s' %(p, average) )
            plt.savefig("p%.1f_%s.png" % (p,average))
            #plt.show()
            
