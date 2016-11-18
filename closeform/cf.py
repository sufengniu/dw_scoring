import scipy.io
import math
import numpy as np
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, find
from scipy.sparse import eye
from scipy import sparse
from numpy.linalg import inv
import time
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from scoring import score

import sys
import getopt
import os, struct

def usage():
    print '''
        -m, --mat: input mat file
        -i, --inv: method use matrix inversion
        -n, --network: mat file network name
        -g, --group: mat file group name
        -a, --alpha: alpha value
        -l, --L: L value, relation with alpha
        -s, --size: representation size
        -t, --iter: number of iterations for SVD
        -h, --help: help function
        '''

method_inv = False
mat_file = 'blogcatalog.mat'
n_name = 'network'
g_name = 'group'

alpha = 0.85
L = 40
size = 64
iters = 30

try:
    opts, args = getopt.getopt(sys.argv[1:], "hm:in:g:a:l:s:t", ["file="])
except getopt.GetoptError, err:
    print 'invalid command line'
    usage()
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-m", "--mat"):
        mat_file = arg
    if opt in ("-i", "--inv"):
        method_inv = True
    if opt in ("-g", "--group"):
        g_name = arg
    if opt in ("-n", "--network"):
        n_name = arg
    if opt in ("-a", "--alpha"):
        alpha = float(arg)
    if opt in ("-l", "--L"):
        L = int(arg)
    if opt in ("-s", "--size"):
        size = int(arg)
    if opt in ("-t", "--iter"):
        iterst = int(arg)
    elif opt in ("-h", "--help"):
        usage()

mat = scipy.io.loadmat(mat_file)
A = mat.get(n_name)
Meta = mat.get(g_name)
n = A.shape[0]
d = np.squeeze(np.asarray(A.sum(axis=1)))
[rows, columns, value] = find(A)
P = csr_matrix((value/(1e-64 + d[rows]), (rows, columns)), shape=(n, n))

if not method_inv:
    # -- method 1
    H = [];

    # finite-step random walk without memory
    Pi = P
    tmp = P
    for step in range(0, L):
        tmp = P.dot(tmp)
        Pi = Pi + tmp;
    Y = np.log(Pi.toarray())

else:
    # -- method 2
    Pinv = inv( (csr_matrix(eye(n))-csr_matrix.multiply(P, alpha)).toarray() )
    Pi = np.divide( Pinv-csr_matrix(eye(n)), alpha )
    Y = np.log(Pi)


U, Sigma, VT = randomized_svd(Y, n_components=size, n_iter= iters, random_state=None)
Uw = U.dot( np.diag( np.sqrt(Sigma)) )

print "===================================================="
print "result without top-k"
score(Uw, startfrom0=True, topk=True)
print "result with top-k"
score(Uw, startfrom0=True, topk=False)
print "===================================================="
print "\n\n"











