from utils import *

import time
import scipy

import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

filename = 'cora'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(filename) # 'cora', 'citeseer', 'pubmed'

scipy.io.savemat(filename, mdict={'network': adj.tocsc()})
features = preprocess_features(features)
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'blogcatalog.mat', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for prop_model: ' + str(FLAGS.prop_model))

import numpy
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

#from utils import TopKRanker, sparse2graph, label2onehot


s0 = True
top_k = False

embeddings_file = 'cora_infinite_poly_4.emb'

A = adj.tocsc()
graph = sparse2graph(A)
labels_matrix = y_train

index_align = 0

embed = numpy.loadtxt(embeddings_file, skiprows=1)
features_matrix = numpy.asarray([embed[numpy.where(embed[:,0]==node+index_align), 1:][0,0] for node in range(A.shape[0])])
features_matrix = numpy.reshape(features_matrix, [features_matrix.shape[0], features_matrix.shape[-1]])

features=preprocess_features(sp.lil_matrix(features_matrix.astype(np.float32)))
# if classifier == 'log':
#     clf = LogisticRegression(max_iter=500,)
# elif classifier == 'svm':
#     clf = LinearSVC()

# clf.fit(X_train, y_train)

placeholders = {
    # 'prediction': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = GCN(placeholders, input_dim=features[2][1], logging=True)
sess=tf.InteractiveSession()
#sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.initialize_all_variables())

cost_val = []

# Train model
for epoch in range(1000):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #    print("Early stopping...")
    #    break

print("Optimization Finished!")


