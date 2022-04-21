from examples import EXAMPLE_001, EXAMPLE_002
import numpy as np



# deflated MLMC
# Schwinger 16^2

def G101():
    # this example computes the chosen matrix via MLMC
    params = set_params('schwinger16')

    # fixed params
    params['spec_function'] = 'mg'
    params['function_tol'] = 1e-3
    params['function_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_002(params)

# deflated Hutchinson
# Schwinger 16^2

def G201():
    # this example computes the chosen matrix via deflated Hutchinson
    params = set_params('schwinger16')

    # fixed parameters
    params['spec_function'] = 'mg'
    params['function_tol'] = 1e-3
    params['function_lambda_min'] = 1e-2
    params['aggregation_type'] = 'XX'

    EXAMPLE_001(params)

# -------------------------------------------------------------

def set_params(example_name):

    if example_name=='schwinger16':

        np.random.seed(51234)

        params = dict()
        matrix_params = dict()

        # to modify
        params['trace_tol'] = 1.0e-1
        params['trace_use_Q'] = False
        params['max_nr_levels'] = 3
        params['coarsest_level_directly'] = True
        # 'high' : 1.0e-9
        # 'low'  : 1.0e-3
        params['accuracy_mg_eigvs'] = 'high'
        params['nr_deflat_vctrs'] = 0

        #matrix_params['mass'] = -0.8940
        #matrix_params['mass'] = -0.95
        # 1.00690114
        #matrix_params['mass'] = -1.006
        matrix_params['mass'] = -0.6

        params['aggrs'] = [2*2,2*2,2*2]
        params['dof'] = [2,2,2,2]

        params['function'] = 'inverse'

        # fixed parameters
        params['trace_multilevel_construction'] = 'manual_aggregation'
        matrix_params['problem_name'] = 'schwinger'
        params['matrix'] = 'schwinger16.mat'
        matrix_params['N'] = 16
        params['matrix_params'] = matrix_params

        return params

    else:
        raise Exception("Non-existent option for example type.")