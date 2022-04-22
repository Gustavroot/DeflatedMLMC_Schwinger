# Some extra utils functions


import multigrid as mg

def flopsV(nr_levels, levels_info, level_id):
    #if nr_levels == 1:
    #    return levels_info[0].A.nnz
    #elif level_id == nr_levels-2:
    #    return 2 * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    #else:
    #    return 2*levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

    # adds the Krylov that wraps the AMG solver
    if nr_levels == 1:
        return levels_info[0].A.nnz
    elif level_id == nr_levels-2:
        #FIXME
        return (2+2) * levels_info[level_id].A.nnz + levels_info[level_id+1].A.nnz
    else:
        # FIXME
        if level_id==0:
            return (2+2) * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)
        else:
            return (2+2) * levels_info[level_id].A.nnz + flopsV(nr_levels, levels_info, level_id+1)

# adds the Krylov that wraps the AMG solver
def flopsV_manual(bare_level, levels_info, level_id):
    #if bare_level==(len(levels_info)-1):
    #    return mg.coarsest_iters_avg * levels_info[0].A.nnz
    #elif level_id == nr_levels-2:
    if level_id == len(levels_info)-2:
        #return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + mg.coarsest_iters_avg*levels_info[level_id+1].A.nnz
        # FIXME : number 30 hardcoded
        #return 2 * mg.smoother_iters * levels_info[level_id].A.nnz + 30*levels_info[level_id+1].A.nnz
        if level_id==bare_level:
            return (2 * mg.smoother_iters + 2) * levels_info[level_id].A.nnz + 0
        else:
            return (2 * mg.smoother_iters + 1) * levels_info[level_id].A.nnz + 0
    else:
        if level_id==bare_level:
            return (2 * mg.smoother_iters + 2) * levels_info[level_id].A.nnz + flopsV_manual(bare_level, levels_info, level_id+1)
        else:
            return (2 * mg.smoother_iters + 1) * levels_info[level_id].A.nnz + flopsV_manual(bare_level, levels_info, level_id+1)








def print_post_results(A,params,result,example):

    if example=="mlmc":
        print(" -- matrix : "+params['matrix'])
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

    elif example=="hutchinson":
        print(" -- matrix : "+params['matrix'])
        print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
        print(" -- tr(A^{-1}) = "+str(result['trace']))
        cmplxity = result['total_complexity']/(1.0e+6)
        print(" -- total MG complexity = "+str(cmplxity)+" MFLOPS")
        print(" -- std dev = "+str(result['std_dev']))
        print(" -- var = "+str(result['std_dev']*result['std_dev']))
        print(" -- number of estimates = "+str(result['nr_ests']))
        print(" -- function iters = "+str(result['function_iters']))

    else:
        raise Exception("Value for parameter <example> not available.")






def trace_params_from_params(params,example):

    if example=="mlmc":
        trace_params = dict()
        function_params = dict()
        function_params['tol'] = params['function_tol']
        trace_params['function_params'] = function_params
        trace_params['tol'] = params['trace_tol']
        trace_params['max_nr_ests'] = 1000000
        trace_params['max_nr_levels'] = params['max_nr_levels']
        trace_params['problem_name'] = params['matrix_params']['problem_name']
        trace_params['coarsest_level_directly'] = params['coarsest_level_directly']
        trace_params['accuracy_mg_eigvs'] = params['accuracy_mg_eigvs']
        trace_params['aggrs'] = params['aggrs']
        trace_params['dof'] = params['dof']
        return trace_params

    elif example=="hutchinson":
        trace_params = dict()
        function_params = dict()
        function_params['tol'] = params['function_tol']
        trace_params['function_params'] = function_params
        trace_params['tol'] = params['trace_tol']
        trace_params['max_nr_ests'] = 100000
        trace_params['max_nr_levels'] = params['max_nr_levels']
        trace_params['problem_name'] = params['matrix_params']['problem_name']
        trace_params['nr_deflat_vctrs'] = params['nr_deflat_vctrs']
        trace_params['accuracy_mg_eigvs'] = params['accuracy_mg_eigvs']
        trace_params['aggrs'] = params['aggrs']
        trace_params['dof'] = params['dof']
        return trace_params

    else:
        raise Exception("Value for parameter <example> not available.")
