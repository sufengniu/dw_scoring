import numpy as np
import cPickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy
from scipy.io import loadmat
from sklearn.utils import shuffle

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
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
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
       out.append(np.sum(np.eye(classes)[i], axis=0))
    return np.array(out)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def multi_sample_mask(idx, l):
    """Create mask."""
    idx=idx.astype(int)
    mask = np.zeros(l)
    for i in range(idx.shape[1]):
        mask[idx[:,i],i] = 1
    return np.array(mask, dtype=np.bool)


# def balanced_subsample(label, percent):

#     train_size = int(label.shape[0]*percent)
#     test_size = label.shape[0]-train_size
#     x_train = np.zeros([train_size, label.shape[1]])
#     y_train = np.zeros([train_size, label.shape[1]])
#     #x_test = np.zeros([test_size, label.shape[1]])
#     #y_test = np.zeros([test_size, label.shape[1]])

#     y = label
#     subsample_size = percent

#     for i in range(y.shape[1]):
#         y = label[:,i]
#         x = np.array(range(y.shape[0]))
#         class_xs = []
#         min_elems = None

#         for yi in np.unique(y):
#             elems = x[(y == yi)]
#             class_xs.append((yi, elems))
#             if min_elems == None or elems.shape[0] < min_elems:
#                 min_elems = elems.shape[0]

#         # half one and half zero
#         train_batch = train_size // (2*int(min_elems*percent))

#         use_elems = min_elems
#         if subsample_size < 1:
#             use_elems = int(min_elems*subsample_size)

#         xs = []
#         ys = []
#         xts = []
#         yts = []

#         for ci,this_xs in class_xs:
#             if len(this_xs) > use_elems:
#                 np.random.shuffle(this_xs)

#             if ci == 0.0:
#                 x_ = this_xs[:(train_size-use_elems*train_batch)]
#                 xt_ = this_xs[(train_size-use_elems*train_batch):]
#             else:
#                 x_ = np.tile(this_xs[:use_elems], train_batch)
#                 xt_ = this_xs[use_elems:]

#             y_ = np.empty(x_.shape)
#             y_.fill(ci)

#             yt_ = np.empty(xt_.shape)
#             yt_.fill(ci)

#             xs.append(x_)
#             ys.append(y_)
#             xts.append(xt_)
#             yts.append(yt_)

#         xs = np.concatenate(xs)
#         ys = np.concatenate(ys)
#         xts = np.concatenate(xts)
#         yts = np.concatenate(yts)
#         xs, ys = shuffle(xs, ys, random_state=0)

#         x_train[:,i] = xs
#         y_train[:,i] = ys
#         #x_test[:,i] = xts
#         #y_test[:,i] = yts

#     return x_train, y_train

def balanced_subsample(label, percent):

    y_train = np.zeros(label.shape)
    train_mask = np.zeros(label.shape)
    #x_test = np.zeros([test_size, label.shape[1]])
    #y_test = np.zeros([test_size, label.shape[1]])

    y = label
    subsample_size = percent

    for i in range(y.shape[1]):
        y = label[:,i]
        x = np.array(range(y.shape[0]))
        class_xs = []
        min_elems = None

        for yi in np.unique(y):
            elems = x[(y == yi)]
            class_xs.append((yi, elems))
            if min_elems == None or elems.shape[0] < min_elems:
                min_elems = elems.shape[0]

        # half one and half zero
        use_elems = min_elems
        if subsample_size < 1:
            use_elems = int(min_elems*subsample_size)

        xs = []
        ys = []
        xts = []
        yts = []

        for ci,this_xs in class_xs:
            if len(this_xs) > use_elems:
                np.random.shuffle(this_xs)

            x_ = this_xs[:use_elems]
            xt_ = this_xs[use_elems:]

            y_ = np.empty(x_.shape)
            y_.fill(ci)

            yt_ = np.empty(xt_.shape)
            yt_.fill(ci)

            xs.append(x_)
            ys.append(y_)
            xts.append(xt_)
            yts.append(yt_)

        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        xts = np.concatenate(xts)
        yts = np.concatenate(yts)
        xs, ys = shuffle(xs, ys, random_state=0)

        train_mask[xs,i] = 1
        y_train[xs,i] = ys
        #x_test[:,i] = xts
        #y_test[:,i] = yts

    return y_train, np.array(train_mask, dtype=np.bool)


def load_mat_data(dataset_str, train_percent, adj_name='network', label_name='group', attr_enable=False):

    if dataset_str == 'blogcatalog.mat':
        dataset_str = "data/{}".format(dataset_str)
        mat = scipy.io.loadmat(dataset_str)
        adj = mat.get(adj_name).tocsr()
        labels = mat.get(label_name)
        features = sp.eye(adj.shape[0]).tolil()
        y_train, train_mask = balanced_subsample(labels.toarray(), 0.5)

        val_range = range(labels.shape[0])
        test_range = range(labels.shape[0])

        val_mask = sample_mask(val_range, labels.shape[0])
        test_mask = sample_mask(test_range, labels.shape[0])

        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_val[val_mask, :] = labels.toarray()[val_mask, :]
        y_test[test_mask, :] = labels.toarray()[test_mask, :]
        return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask
    else:
        dataset_str = "data/{}".format(dataset_str)
        mat = scipy.io.loadmat(dataset_str)
        adj = mat.get(adj_name).tocsr()
        labels = mat.get(label_name)

        if attr_enable:
            features = mat.get('location_id')
            features = sp.hstack((features, mat.get('education')))
            #features = mat.get('education')
            
            #for key, value in mat.iteritems():
            #    if not (key.startswith('__') or key.startswith(adj_name) or key.startswith(adj_name)):
            #        attr = mat.get(key)
            #        features.append(attr)
            #    for i in features:
            #        features = sp.hstack((features, i))
        else:
            features = sp.eye(adj.shape[0])

        features = features.tolil()

        train_range = range(int(labels.shape[0]*train_percent))
        val_range = range(int(labels.shape[0]*train_percent), int(labels.shape[0] * train_percent)+20)
        test_range = range(int(labels.shape[0]*train_percent), labels.shape[0])

        train_mask = sample_mask(train_range, labels.shape[0])
        val_mask = sample_mask(val_range, labels.shape[0])
        test_mask = sample_mask(test_range, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels.toarray()[train_mask, :]
        y_val[val_mask, :] = labels.toarray()[val_mask, :]
        y_test[test_mask, :] = labels.toarray()[test_mask, :]
    	return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    #feed_dict.update({placeholders['prediction']: prediction})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
