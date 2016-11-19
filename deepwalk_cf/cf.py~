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
from smart_open import smart_open

import sys
import getopt
import os, struct

def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8

def save_word2vec_format(fname, embedding, binary=False):
    """
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
     `fname` is the file used to save the vectors in
     `fvocab` is an optional file used to save the vocabulary
     `binary` is an optional boolean indicating whether the data is to be saved
     in binary word2vec format (default: False)
    """
    with smart_open(fname, 'wb') as fout:
        fout.write(to_utf8("%s %s\n" % embedding.shape))
        # store in sorted order: most frequent words at the top
        for word in range(embedding.shape[0]):
            row = embedding[word]
            if binary:
                fout.write(to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

def usage():
    print '''
        -m: input mat file
        -i: method use matrix inversion, default is false
        -n: mat file network name
        -g: mat file group name
        -a: alpha value, default is 0.85
        -l: L value, has relation with alpha, default is 7
        -s: representation size
        -t: number of iterations for SVD
        -o: output embedding file name
        -h: help function
        '''

method_inv = False
mat_file = 'blogcatalog.mat'
n_name = 'network'
g_name = 'group'

alpha = 0.85
L = 7
size = 64
iters = 30

#try:
opts, args = getopt.getopt(sys.argv[1:], "s:m:ig:n:a:l:t:ho:", ["file="])
#except getopt.GetoptError, err:
#    print 'invalid command line'
#    usage()
#    sys.exit(2)
for opt, arg in opts:
    if opt in ("-m"):
        mat_file = arg
    if opt in ("-i"):
        method_inv = True
    if opt in ("-g"):
        g_name = arg
    if opt in ("-n"):
        n_name = arg
    if opt in ("-a"):
        alpha = float(arg)
    if opt in ("-l"):
        L = int(arg)
    if opt in ("-s"):
        size = int(arg)
    if opt in ("-t"):
        iters = int(arg)
    if opt in ("-o"):
        output_file = arg
    elif opt in ("-h"):
        usage()
        sys.exit()

mat = scipy.io.loadmat(mat_file)
A = mat.get(n_name)
Meta = mat.get(g_name)
n = A.shape[0]
d = np.squeeze(np.asarray(A.sum(axis=1)))
[rows, columns, value] = find(A)
P = csr_matrix((value/(1e-64 + d[rows]), (rows, columns)), shape=(n, n))

print "loading input file done, start computing..."

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

print "calculating SVD"

U, Sigma, VT = randomized_svd(Y, n_components=size, n_iter= iters, random_state=None)
Uw = U.dot( np.diag( np.sqrt(Sigma)) )

save_word2vec_format(output_file, Uw)

'''
print "===================================================="
print "result without top-k"
score(Uw, startfrom0=True, topk=True)
print "result with top-k"
score(Uw, startfrom0=True, topk=False)
print "===================================================="
print "\n"
'''










