from matrix import loadMatrix
from function import loadFunction
import scipy as sp
import numpy as np
from stoch_trace import hutchinson, mlmc

import time



# this example assumes the matrix is Hermitian, and computes via deflated Hutchinson
def EXAMPLE_001(params):

    print("\n----------------------------------------------------------")
    print("Example 01 : computing tr(A^{-1}) with deflated Hutchinson")
    print("----------------------------------------------------------\n")

    # checking input params
    if 'matrix' not in params:
        raise Exception('From <EXAMPLE_001(...)> : specify a matrix type.')
    else:
        matrix_name = params['matrix']

    if 'function' not in params:
        # choosing cg as default
        spec_name = 'cg'
    else:
        spec_name = params['spec_function']
    if 'matrix_params' not in params:
        raise Exception("From <EXAMPLE_001(...)> : you need to provide the params of the matrix.")

    # extracting matrix
    A,B = loadMatrix(matrix_name, params['matrix_params'])

    # setting function
    function_name = params['function']
    function = loadFunction(spec_name, A=A, B=B, function_name=function_name)

    # TODO : check more input params

    trace_tol = params['trace_tol']
    trace_use_Q = params['trace_use_Q']
    function_tol = params['function_tol']
    max_nr_levels = params['max_nr_levels']

    trace_params = dict()
    function_params = dict()
    function_params['spec_name'] = spec_name
    function_params['function_name'] = function_name
    function_params['tol'] = function_tol
    trace_params['function_params'] = function_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 1000000
    trace_params['use_Q'] = trace_use_Q
    trace_params['max_nr_levels'] = max_nr_levels
    trace_params['problem_name'] = params['matrix_params']['problem_name']
    trace_params['nr_deflat_vctrs'] = params['nr_deflat_vctrs']
    #trace_params['aggregation_type'] = params['aggregation_type']
    trace_params['accuracy_mg_eigvs'] = params['accuracy_mg_eigvs']
    trace_params['aggrs'] = params['aggrs']
    trace_params['dof'] = params['dof']
    trace_params['function'] = params['function']
    #start = time.time()
    result = hutchinson(A, function, trace_params)
    #end = time.time()
    #print("Total Hutchinson time = "+str(end-start)+" cpu seconds")
    trace = result['trace']
    std_dev = result['std_dev']
    nr_ests = result['nr_ests']
    function_iters = result['function_iters']

    if function_name=="inverse" and spec_name=='mg':
        total_complexity = result['total_complexity']

    print(" -- matrix : "+matrix_name)
    print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
    print(" -- tr(A^{-1}) = "+str(trace))
    if function_name=="inverse" and spec_name=='mg':
        cmplxity = total_complexity/(1.0e+6)
        print(" -- total MG complexity = "+str(cmplxity)+" MFLOPS")
    print(" -- std dev = "+str(std_dev))
    print(" -- var = "+str(std_dev*std_dev))
    print(" -- number of estimates = "+str(nr_ests))
    print(" -- function iters = "+str(function_iters))

    print("\n")

    if function_name=="exponential":
        del function


# this example assumes the matrix is Hermitian, and computes via MLMC
def EXAMPLE_002(params):

    print("\n-------------------------------------------")
    print("Example 02 : computing tr(A^{-1}) with MLMC")
    print("-------------------------------------------\n")

    # checking input params
    if 'matrix' not in params:
        raise Exception('From <EXAMPLE_002(...)> : specify a matrix type.')
    else:
        matrix_name = params['matrix']
    if 'function' not in params:
        # choosing cg as default
        spec_name = 'cg'
    else:
        spec_name = params['spec_function']
    if 'matrix_params' not in params:
        raise Exception("From <EXAMPLE_002(...)> : you need to provide the params of the matrix.")

    # extracting matrix
    A,B = loadMatrix(matrix_name, params['matrix_params'])

    # setting function
    function_name = params['function']
    function = loadFunction(spec_name, A=A, B=B, function_name=function_name)

    # TODO : check more input params

    trace_tol = params['trace_tol']
    trace_use_Q = params['trace_use_Q']
    function_tol = params['function_tol']
    function_lambda_min = params['function_lambda_min']
    max_nr_levels = params['max_nr_levels']
    trace_ml_constr = params['trace_multilevel_construction']

    trace_params = dict()
    function_params = dict()
    function_params['spec_name'] = spec_name
    function_params['function_name'] = function_name
    function_params['tol'] = function_tol
    function_params['lambda_min'] = function_lambda_min
    trace_params['function_params'] = function_params
    trace_params['tol'] = trace_tol
    trace_params['max_nr_ests'] = 1000000
    trace_params['max_nr_levels'] = max_nr_levels
    trace_params['multilevel_construction'] = trace_ml_constr
    trace_params['use_Q'] = trace_use_Q
    trace_params['problem_name'] = params['matrix_params']['problem_name']
    trace_params['aggregation_type'] = params['aggregation_type']
    trace_params['coarsest_level_directly'] = params['coarsest_level_directly']
    trace_params['accuracy_mg_eigvs'] = params['accuracy_mg_eigvs']
    trace_params['aggrs'] = params['aggrs']
    trace_params['dof'] = params['dof']
    trace_params['function'] = params['function']
    #start = time.time()
    result = mlmc(A, function, trace_params)
    #end = time.time()
    #print("Total MLMC time = "+str(end-start)+" cpu seconds")

    print(" -- matrix : "+matrix_name)
    print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
    print(" -- tr(A^{-1}) = "+str(result['trace']))
    cmplxity = result['total_complexity']/(1.0e+6)
    print(" -- total MG complexity = "+str(cmplxity)+" MFLOPS")
    #print(" -- std dev = "+str(result['std_dev']))
    print(" -- std dev = ---")
    for i in range(result['nr_levels']):
        print(" -- level : "+str(i))
        print(" \t-- number of estimates = "+str(result['results'][i]['nr_ests']))
        print(" \t-- function iters = "+str(result['results'][i]['function_iters']))
        print(" \t-- trace = "+str(result['results'][i]['ests_avg']))
        print(" \t-- std dev = "+str(result['results'][i]['ests_dev']))
        print(" \t-- var = "+str(result['results'][i]['ests_dev'] * result['results'][i]['ests_dev']))
        #if i<(result['nr_levels']-1):
        cmplxity = result['results'][i]['level_complexity']/(1.0e+6)
        print("\t-- level MG complexity = "+str(cmplxity)+" MFLOPS")

    print("\n")

    if function_name=="exponential":
        del function
