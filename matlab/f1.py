import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle

from collections import defaultdict

import sys

def scoring(y_test, preds, tp):
    all_results = defaultdict(list)
    training_percents = [tp]
    for training_percent in training_percents:
        results = {}
        averages = ["micro", "macro", "samples", "weighted"]

        for average in averages:
            results[average] = f1_score(y_test, preds, average=average)
        
        all_results[training_percent].append(results)

    print 'Results, using embeddings of dimensionality'
    print '-------------------'
    for train_percent in sorted(all_results.keys()):
        print 'Train percent:', train_percent
        for x in all_results[train_percent]:
            print  x
        print '-------------------'

