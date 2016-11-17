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

method_inv = True

mat = scipy.io.loadmat('blogcatalog.mat')
A = mat.get('network')
Meta = mat.get('group')
n = A.shape[0]
d = np.squeeze(np.asarray(A.sum(axis=1)))
[rows, columns, value] = find(A)
P = csr_matrix((value/(1e-64 + d[rows]), (rows, columns)), shape=(n, n))

alpha = 0.85
L = 40;

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


U, Sigma, VT = randomized_svd(Y, n_components=64, n_iter= 30, random_state=None)
Uw = U.dot( np.diag( np.sqrt(Sigma)) )

score(Uw, startfrom0=True, topk=False)

score(Uw, startfrom0=True, topk=False)












