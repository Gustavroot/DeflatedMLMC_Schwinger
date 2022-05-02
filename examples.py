from matrix import loadMatrix
import scipy as sp
import numpy as np
from stoch_trace import hutchinson,mlmc
from utils import print_post_results,trace_params_from_params

import time




# this example assumes the matrix is Hermitian, and computes via deflated Hutchinson
def EXAMPLE_001(params):

    print("\n----------------------------------------------------------")
    print("Example 01 : computing tr(A^{-1}) with deflated Hutchinson")
    print("----------------------------------------------------------\n")

    # extracting matrix
    A = loadMatrix(params['matrix'], params['matrix_params'])

    trace_params = trace_params_from_params(params,"hutchinson")

    start = time.time()
    result = hutchinson(A, trace_params)
    end = time.time()
    print("Total Hutchinson time = "+str(end-start)+" cpu seconds\n")

    print_post_results(A,params,result,"hutchinson")




# this example assumes the matrix is Hermitian, and computes via MLMC
def EXAMPLE_002(params):

    print("\n-------------------------------------------")
    print("Example 02 : computing tr(A^{-1}) with MLMC")
    print("-------------------------------------------\n")

    # extracting matrix
    A = loadMatrix(params['matrix'], params['matrix_params'])

    trace_params = trace_params_from_params(params,"mlmc")

    start = time.time()
    result = mlmc(A, trace_params)
    end = time.time()
    print("Total MLMC time = "+str(end-start)+" cpu seconds")

    print_post_results(A,params,result,"mlmc")
