
from __future__ import division

import glob
import sys
import os
from os.path import basename
import scipy.io
import math
import numpy as np
import scipy
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, find
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy import sparse
from numpy.linalg import inv
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from itertools import izip
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from numpy import linalg as LA

from collections import defaultdict
from scipy.sparse import csr_matrix,vstack,hstack


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}

def preprocessing(shuf, train_percent):
    X, y = shuf
    training_size = int(train_percent * X.shape[0])
    X_train = X[:training_size, :]
    y_train_ = y[:training_size]
    y_train = y_train_.toarray()
    y_train = np.reshape(y_train, (-1))

    '''
    y_train = [[] for x in xrange(y_train_.shape[0])]
    cy =  y_train_.tocoo()
    for i, j in izip(cy.row, cy.col):
        y_train[i].append(j)
    '''
    X_test = X[training_size:, :]
    y_test_ = y[training_size:]
    y_test = y_test_.toarray()
    y_test = np.reshape(y_test, (-1))
    '''
    y_test = [[] for x in xrange(y_test_.shape[0])]
    cy =  y_test_.tocoo()
    for i, j in izip(cy.row, cy.col):
        y_test[i].append(j)
    '''
    return X_train, y_train, X_test, y_test

def prediction(X_train, y_train, X_test, y_test, classifier='log'):
    if classifier == 'log':
        clf = LogisticRegression(max_iter=500,)
    elif classifier == 'svm':
        clf = LinearSVC()

    if np.sum(y_train) == 0.0:
        preds = np.zeros(X_test.shape[0])
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
    return preds

def score(preds, label):
    results = {}
    averages = ["micro", "macro", "samples", "weighted"]
    for average in averages:
        results[average] = f1_score(label,  preds, average=average)
    return results

# template
def buildAttributeGraph(X_select,PG):
    n,T = X_select.shape
    P_Ai = np.zeros([n, n, T])
    for i_attribute in range(T):
        P_Ai[:,:,i_attribute] = PG.toarray()
        tmp = X_select[:,i_attribute].dot(  csr_matrix.multiply( csc_matrix.transpose(X_select[:,i_attribute]), 1/(1e-64 + X_select[:,i_attribute].sum())) )
        flyout_ind, cols = X_select[:, i_attribute].nonzero()
        P_Ai[flyout_ind,:,i_attribute] = tmp[flyout_ind, :].toarray()
    return P_Ai


# template
def buildTemplate(group):
    template = (group*np.transpose(group)).toarray()
    ind = find(template == 0)
    template[ind[0],ind[1]] = -1

    for i_row in range(template.shape[0]):
        template[i_row, i_row] = 0

    ind = find(template > 0)
    pos_num = template[ind[0],ind[1]].sum()
    zero_ind = find( group.sum(axis=1) == 0)
    #template[zero_ind,zero_ind] = 0
    for  i_row in range(zero_ind[0].shape[0]):
        for  i_column in range(zero_ind[0].shape[0]):
            template[zero_ind[0][i_row],zero_ind[0][i_column]] = 0

    ind = find(template < 0)
    neg_num = template[ind[0],ind[1]].sum()
    template[ind[0],ind[1]] = pos_num/neg_num   # remove test data (TBD)
    return template


# add up
def mergeTransition(mu, PG, P_Ai, alpha):
    P = np.multiply(PG,(1- mu.sum()))
    n,m,T = P_Ai.shape
    for i_attribute in range(T):
        P = P + np.multiply( P_Ai[:,:,i_attribute], mu[i_attribute])

    Pinv = inv( eye(n) - np.multiply(P, alpha) )  # Pinv = (I-\alpha*P)
    Pi = np.multiply( Pinv - eye(n), 1.0/alpha )  # (Pinv -I)/alpha
    ind = find( Pi < 1e-16 )
    Pi[ind[0],ind[1]] = 1e-16;
    Y = np.log(Pi)
    return Y

def jaccard_sim(x,y):
    obj_val = np.multiply(x,y).sum() / ( 1e-16 + x.sum() + y.sum() - np.multiply(x,y).sum() )
    return obj_val

def max_fit(group, X ):
    n,M = group.shape
    n,T = X.shape

    coding = np.zeros([T])
    y = group.toarray()

    for i_attribute in range(T):
        coding[i_attribute] = jaccard_sim(y, X[:,i_attribute].toarray())
    
    index = np.where( coding == np.max(coding))

    return index[0][0]

def cover_fit(group, X, K):
    n,M = group.shape
    n,T = X.shape

    coding = np.zeros([T,M])
    for i_group in range(M):

        y = group[:,i_group].toarray()
        for i_attribute in range(T):
            coding[i_attribute, i_group] = jaccard_sim(y, X[:,i_attribute].toarray())
        coding_sort = sorted(coding[:,i_group],reverse=True)
        ind = find(coding[:,i_group] < coding_sort[min(K,T-1) ])
        coding[ind[1],i_group] = 0;
    if coding.sum() > 0:
       coding_ind = coding.sum(axis=1)
       index = find(coding_ind > 0)[1]
    else:
       index = 0

    X_select = X[:,index]
    return X_select, index

def logic_fit(group, X, K):
    n,M = group.shape
    n,T = X.shape

    coding = np.zeros([T,M])
    for i_group in range(M):
        y = group[:,i_group].toarray()

        x_aug = np.zeros([n,1])
        Corr = np.zeros([T,1])
        for i_k in range(K):
            for i_attribute in range(T):
                tmp = x_aug + X[:,i_attribute]
                ind = find(tmp > 1)
                tmp[ind[0],ind[1]] = 1
                Corr[i_attribute,0] = jaccard_sim(y, tmp)

            index = np.argmax(Corr)
            tmp = x_aug + X[:,index]
            ind = find(tmp > 1)
            tmp[ind[0],ind[1]] = 1

            if jaccard_sim(y, tmp) > jaccard_sim(y, x_aug):
                x_aug = tmp
                coding[index, i_group] = 1

    coding_ind = coding.sum(axis=1)
    z = find(coding_ind > 0)
    X_select = X[:,z[1]]
    return X_select


def attributeEmbedding(PG, X, group, option):
    alpha = option['flyout']
    dimension = option['dimension']
    sparsity = option['sparsity']

    # thr_ind = 5;
    mag_set = np.array([0, 0.1, 0.3, 0.5, 0.8, 0.99])
    if option['overlap']:
        X_select, index = cover_fit(group, X, sparsity)
    else:
        X_select = logic_fit(group, X, sparsity)

    if X_select.shape[1] != 0:

        template = buildTemplate(group)
        P_Ai = buildAttributeGraph(X_select,PG)
        n, T = X_select.shape
        mu = np.zeros([T,1])
        for i_attribute in range(T):
            mu[i_attribute] = np.multiply( template, P_Ai[:,:,i_attribute] - PG).sum()
        mu_sort = sorted(mu,reverse=True)


        # ind = find(mu < max(mu_sort[thr_ind], 0))
        ind = find(mu < 0)
        mu[ind[0],ind[1]] = 0
        if mu.sum() > 0:
            mu = np.multiply( mu, 1.0/(1e-64 + mu.sum()))
            obj_value = np.zeros([ mag_set.shape[0],1])
            for i_mag in range(mag_set.shape[0]):
                Y = mergeTransition( np.multiply(mag_set[i_mag],mu), PG, P_Ai, alpha)
                obj_value[i_mag,0] = np.multiply( template, Y ).sum()

            index = np.argmax(obj_value)
            mu = np.multiply(mag_set[index],mu)

        Y = mergeTransition( mu, PG, P_Ai, alpha)
        U, Sigma, VT = randomized_svd(Y, n_components=dimension, n_iter= 30, random_state=None)
        Uw = U.dot( np.diag( np.sqrt(Sigma)) )

    else:
        Uw = deepwalk_infty_Embedding(PG, option)
        mu = 0
    return Uw, mu


def  deepwalk_infty_Embedding(PG, option):
    alpha = option['flyout']
    dimension = option['dimension']
    Pinv = inv( (csr_matrix(eye(n)) - csr_matrix.multiply(PG, alpha)).toarray() )  # Pinv = (I-\alpha*P)
    Pi = np.divide(  Pinv - csr_matrix(eye(n)), alpha )  # (Pinv -I)/alpha
    ind = find( Pi < 1e-16 )
    Pi[ind[0],ind[1]] = 1e-16
    Y = np.log(Pi)
    U, Sigma, VT = randomized_svd(Y, n_components=dimension, n_iter= 30, random_state=None)
    Uw = U.dot( np.diag( np.sqrt(Sigma)) )
    return Uw

def  deepwalk_fty_Embedding(PG, option):
    L = option['step']
    dimension = option['dimension']
    Pi = PG
    tmp = PG
    for step in range(0, L):
        tmp = PG.dot(tmp)
        Pi = Pi + tmp;
    ind = find( Pi < 1e-16 )
    Pi[ind[0],ind[1]] = 1e-16
    Y = np.log(Pi.toarray())
    U, Sigma, VT = randomized_svd(Y, n_components=dimension, n_iter= 30, random_state=None)
    Uw = U.dot( np.diag( np.sqrt(Sigma)) )
    return Uw

# label = group[:,0]
# option['sparsity'] = 20
# option['overlap'] = 0
# embedding1, mu = attributeEmbedding(PG, X, label, option)
# option['sparsity'] = 100
# option['overlap'] = 1
# embedding3, mu = attributeEmbedding(PG, X, label, option)
# LA.norm(embedding1-embedding3)

# embedding2 = deepwalk_fty_Embedding(PG, option)
# embedding3 = deepwalk_infty_Embedding(PG, option)


option={}
option['dimension'] = 15
option['flyout'] = 0.85
option['sparsity'] = 20
option['overlap'] = 0
option['step'] = 7


k=10
training_percents = [0.1, 0.3, 0.5, 0.9]
preds_1 = {}
preds_2 = {}
preds_3 = {}
preds_4 = {}
y_label_1 = {}
y_label_2 = {}
y_label_3 = {}
y_label_4 = {}
attr_index = 0

# load data
list_of_files = glob.glob('../dataset/kaggle/*.mat')
for f in list_of_files:
    mat = scipy.io.loadmat(f)
# mat = scipy.io.loadmat('../dataset/kaggle/8777.mat') # 11186  1968
    A = mat.get('network')
    group = mat.get('location')
    X = mat.get('education')  # education   location_id
    ind = find(X>1)
    X[ind[0],ind[1]] = 1
    n, group_num = group.shape

    d = np.squeeze(np.asarray(A.sum(axis=1)))
    [rows, columns, value] = find(A)
    PG = csr_matrix((value/(1e-64 + d[rows]), (rows, columns)), shape=(n, n))

    embedding1 = deepwalk_infty_Embedding(PG,option)
    embedding4 = X.toarray()

    for i_group in range(group_num):
        print "calculating group ", i_group
        label = group[:,i_group]
        option['overlap'] = 0
        option['sparsity']=20
        embedding2, _= attributeEmbedding(PG, X, label, option)
        option['overlap'] = 1
        option['sparsity']=100
        embedding3, _=attributeEmbedding(PG, X, label, option)

        # crossvalidation
        shuffles1 = []
        shuffles2 = []
        shuffles3 = []
        shuffles4 = []
        number_shuffles = k
        for x in range(number_shuffles):
            shuf_tmp = skshuffle(embedding1, embedding2, embedding3, embedding4, label)
            emb1 = shuf_tmp[0]
            emb2 = shuf_tmp[1]
            emb3 = shuf_tmp[2]
            emb4 = shuf_tmp[3]


            shuffles1.append([emb1, shuf_tmp[4]])
            shuffles2.append([emb2, shuf_tmp[4]])
            shuffles3.append([emb3, shuf_tmp[4]])
            shuffles4.append([emb4, shuf_tmp[4]])

        preds_tmp_1 = defaultdict(list)
        label_tmp_1 = defaultdict(list)
        for train_percent in training_percents:
            for shuf in shuffles1:
                X_train, y_train, X_test, y_test = preprocessing(shuf, train_percent)
                preds_tmp_1[train_percent].append(prediction(X_train, y_train, X_test, y_test, classifier='log'))
                label_tmp_1[train_percent].append(y_test)
        preds_1[i_group] = preds_tmp_1
        y_label_1[i_group] = label_tmp_1

        preds_tmp_2 = defaultdict(list)
        label_tmp_2 = defaultdict(list)
        for train_percent in training_percents:
            for shuf in shuffles2:
                X_train, y_train, X_test, y_test = preprocessing(shuf, train_percent)
                preds_tmp_2[train_percent].append(prediction(X_train, y_train, X_test, y_test, classifier='log'))
                label_tmp_2[train_percent].append(y_test)
        preds_2[i_group] = preds_tmp_2
        y_label_2[i_group] = label_tmp_2

        preds_tmp_3 = defaultdict(list)
        label_tmp_3 = defaultdict(list)
        for train_percent in training_percents:
            for shuf in shuffles3:
                X_train, y_train, X_test, y_test = preprocessing(shuf, train_percent)
                preds_tmp_3[train_percent].append(prediction(X_train, y_train, X_test, y_test, classifier='log'))
                label_tmp_3[train_percent].append(y_test)
        preds_3[i_group] = preds_tmp_3
        y_label_3[i_group] = label_tmp_3

        # directly use attributes
        preds_tmp_4 = defaultdict(list)
        label_tmp_4 = defaultdict(list)
        for train_percent in training_percents:
            for shuf in shuffles4:
                X_train, y_train, X_test, y_test = preprocessing(shuf, train_percent)
                y_train_t = np.reshape(y_train, (y_train.shape[0],1))
                index = max_fit(csc_matrix(y_train_t), csc_matrix(X_train))
                preds_tmp_4[train_percent].append(X_test[:,index])
        preds_4[i_group] = preds_tmp_4
        y_label_4[i_group] = label_tmp_4

    averages = ["micro", "macro", "samples", "weighted"]
    results1_avg = []
    results2_avg = []
    results3_avg = []
    results4_avg = []
    results_avg = []
    results1_var = []
    results2_var = []
    results3_var = []
    results4_var = []
    results_var = []
    for train_percent in training_percents:

        p1 = [preds_1[m][train_percent] for m in range(group.shape[1])]
        p1 = np.transpose(np.array(p1), [1, 0, 2])
        l1 = [y_label_1[m][train_percent] for m in range(group.shape[1])]
        l1 = np.transpose(np.array(l1), [1, 0, 2])

        p2 = [preds_2[m][train_percent] for m in range(group.shape[1])]
        p2 = np.transpose(np.array(p2), [1, 0, 2])
        l2 = [y_label_2[m][train_percent] for m in range(group.shape[1])]
        l2 = np.transpose(np.array(l2), [1, 0, 2])

        p3 = [preds_3[m][train_percent] for m in range(group.shape[1])]
        p3 = np.transpose(np.array(p3), [1, 0, 2])
        l3 = [y_label_3[m][train_percent] for m in range(group.shape[1])]
        l3 = np.transpose(np.array(l3), [1, 0, 2])

        p4 = [preds_4[m][train_percent] for m in range(group.shape[1])]
        p4 = np.transpose(np.array(p4), [1, 0, 2])
        l4 = [y_label_3[m][train_percent] for m in range(group.shape[1])]
        l4 = np.transpose(np.array(l4), [1, 0, 2])

        score_tmp1=[]
        score_tmp2=[]
        score_tmp3=[]
        score_tmp4=[]

        for j in range(number_shuffles):
            score_tmp1.append(score(p1[j,:,:], l1[j,:,:]))
            score_tmp2.append(score(p2[j,:,:], l2[j,:,:]))
            score_tmp3.append(score(p3[j,:,:], l3[j,:,:]))
            score_tmp4.append(score(p4[j,:,:], l4[j,:,:]))

        average_score1 = []
        average_score2 = []
        average_score3 = []
        average_score4 = []
        var_score1 = []
        var_score2 = []
        var_score3 = []
        var_score4 = []
        for average in averages:
            average_score1.append(np.average([score_tmp1[m][average] for m in range(number_shuffles)]))
            average_score2.append(np.average([score_tmp2[m][average] for m in range(number_shuffles)]))
            average_score3.append(np.average([score_tmp3[m][average] for m in range(number_shuffles)]))
            average_score4.append(np.average([score_tmp4[m][average] for m in range(number_shuffles)]))
            var_score1.append(np.var([score_tmp1[m][average] for m in range(number_shuffles)]))
            var_score2.append(np.var([score_tmp2[m][average] for m in range(number_shuffles)]))
            var_score3.append(np.var([score_tmp3[m][average] for m in range(number_shuffles)]))
            var_score4.append(np.var([score_tmp4[m][average] for m in range(number_shuffles)]))

        results1_avg.append(average_score1)
        results2_avg.append(average_score2)
        results3_avg.append(average_score3)
        results4_avg.append(average_score4)
        results1_var.append(var_score1)
        results2_var.append(var_score2)
        results3_var.append(var_score3)
        results4_var.append(var_score4)

    results_avg = np.array([results1_avg, results2_avg, results3_avg, results4_avg])
    results_var = np.array([results1_var, results2_var, results3_var, results4_var])
    fname = os.path.basename(os.path.splitext(f)[0])
    filename = 'attr_%s_result.mat' % fname
    # filename = 'attr_result.mat'
    scipy.io.savemat(filename, mdict={'average': results_avg, 'variance': results_var})


    # average_score1 = {}
    # average_score2 = {}
    # average_score3 = {}
    # average_score4 = {}
    # var_score1 = {}
    # var_score2 = {}
    # var_score3 = {}
    # var_score4 = {}
    # for average in averages:
        # average_score1[average] = np.average([score_tmp1[m][average] for m in range(number_shuffles)])
        # average_score2[average] = np.average([score_tmp2[m][average] for m in range(number_shuffles)])
        # average_score3[average] = np.average([score_tmp3[m][average] for m in range(number_shuffles)])
        # average_score4[average] = np.average([score_tmp4[m][average] for m in range(number_shuffles)])
        # var_score1[average] = np.var([score_tmp1[m][average] for m in range(number_shuffles)])
        # var_score2[average] = np.var([score_tmp2[m][average] for m in range(number_shuffles)])
        # var_score3[average] = np.var([score_tmp3[m][average] for m in range(number_shuffles)])
        # var_score4[average] = np.var([score_tmp4[m][average] for m in range(number_shuffles)])

    # results1_avg.append(average_score1)
    # results2_avg.append(average_score2)
    # results3_avg.append(average_score3)
    # results4_avg.append(average_score4)
    # results1_var.append(var_score1)
    # results2_var.append(var_score2)
    # results3_var.append(var_score3)
    # results4_var.append(var_score4)

    #results1_avg[train_percent] = average_score1
    #results2_avg[train_percent] = average_score2
    #results3_avg[train_percent] = average_score3
    #results4_avg[train_percent] = average_score4
    #results1_var[train_percent] = var_score1
    #results2_var[train_percent] = var_score2
    #results3_var[train_percent] = var_score3
    #results4_var[train_percent] = var_score4

# results_avg.append(results1_avg)
# results_avg.append(results2_avg)
# results_avg.append(results3_avg)
# results_avg.append(results4_avg)
# results_var.append(results1_var)
# results_var.append(results2_var)
# results_var.append(results3_var)
# results_var.append(results4_var)





