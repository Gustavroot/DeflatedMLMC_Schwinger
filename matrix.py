# All the resources needed to load and manipulate matrices
import warnings
from scipy.sparse import identity
from scipy.sparse import csr_matrix
import matplotlib.pylab as plt

import numpy
import scipy as sp
from scipy.sparse.linalg import eigsh




def loadMatrix(matrix_name, params):

    warnings.simplefilter("ignore")
    import scipy.io as sio
    filename = matrix_name.split('_')

    m = params['mass']
    mat_contents = sio.loadmat(matrix_name)
    A = mat_contents['S']

    # this application of g3 is not needed in general !!
    if matrix_name=='schwinger16.mat':
        mat_size = int(A.shape[0]/2)
        A[mat_size:,:] = -A[mat_size:,:]

    A += m*identity(A.shape[0], dtype=A.dtype)

    return A
