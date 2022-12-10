from examples import EXAMPLE_001, EXAMPLE_002
import numpy as np




# deflated Hutchinson
# Schwinger 16^2

def G101():
    # this example computes the chosen matrix via deflated Hutchinson
    params = set_params('schwinger16')

    # fixed parameters
    params['function_tol'] = 1e-12

    EXAMPLE_001(params)



# deflated MLMC
# Schwinger 16^2

def G201():
    # this example computes the chosen matrix via MLMC
    params = set_params('schwinger16')

    # fixed params
    params['function_tol'] = 1e-12

    EXAMPLE_002(params)



# deflated Hutchinson
# Schwinger 128^2

def G102():
    # this example computes the chosen matrix via deflated Hutchinson
    params = set_params('schwinger128')

    # fixed parameters
    params['function_tol'] = 1e-12

    EXAMPLE_001(params)



# deflated MLMC
# Schwinger 128^2

def G202():
    # this example computes the chosen matrix via MLMC
    params = set_params('schwinger128')

    # fixed params
    params['function_tol'] = 1e-12

    EXAMPLE_002(params)



def set_params(example_name):

    if example_name=='schwinger16':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        #params['trace_tol'] = 0.25e-2
        params['trace_tol'] = 1.0e-2
        params['max_nr_levels'] = 3
        params['coarsest_level_directly'] = True
        # 'high' : 1.0e-9
        # 'low'  : 1.0e-3
        params['accuracy_mg_eigvs'] = 'low'
        params['nr_deflat_vctrs'] = 64
        params['mlmc_deflat_vctrs'] = [16,16]
        #params['mlmc_deflat_vctrs'] = [0,0]

        params['mlmc_levels_to_skip'] = [1]

        matrix_params['mass'] = -1.00690114*0.99

        params['aggrs'] = [2*2,2*2,2*2]
        params['dof'] = [2,2,2]

        # fixed parameters
        matrix_params['problem_name'] = 'schwinger'
        params['matrix'] = 'schwinger16.mat'
        params['matrix_params'] = matrix_params

        return params

    elif example_name=='schwinger128':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        #params['trace_tol'] = 0.25e-2
        params['trace_tol'] = 1.0e-2
        params['max_nr_levels'] = 4
        params['coarsest_level_directly'] = True
        # 'high' : 1.0e-9
        # 'low'  : 1.0e-3
        params['accuracy_mg_eigvs'] = 'low'
        params['nr_deflat_vctrs'] = 512
        params['mlmc_deflat_vctrs'] = [0,0,0]
        #params['mlmc_deflat_vctrs'] = [0,0]

        # exact
        params['defl_eigvs_tol'] = 1.0e-9
        params['diff_lev_op_tol'] = 1.0e-12
        params['defl_type'] = "exact"

        # inexact
        #params['defl_eigvs_tol'] = 1.0e-1
        #params['diff_lev_op_tol'] = 1.0e-3
        #params['defl_type'] = "inexact_01" # ---> uses inversions

        #params['defl_type'] = "inexact_02" # ---> uses inversions
        #params['defl_type'] = "inexact_03" # ---> avoids inversions

        params['mlmc_levels_to_skip'] = [1]

        params['use_permuted'] = True
        params['latt_dims'] = [128,128]
        params['x_displacement'] = 2

        #matrix_params['mass'] = -1.00690114*0.99
        #matrix_params['mass'] = -0.1333

        # for the following params:
        # 	m0 = -0.1320
        # 	permuted = True
        # 	x_displacement = 2
        # the <exact> trace is : -8.748242701374695+50.215154098005584j

        # 0.13353501
        matrix_params['mass'] = -0.1320

        params['aggrs'] = [4*4,4*4,4*4]
        params['dof'] = [2,4,4,4]

        # fixed parameters
        matrix_params['problem_name'] = 'schwinger'
        params['matrix'] = 'schwinger128.mat'
        params['matrix_params'] = matrix_params

        return params

    else:
        raise Exception("Non-existent option for example type.")
