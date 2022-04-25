# Some extra utils functions

import multigrid as mg
import time
from scipy.sparse.linalg import svds,eigsh,eigs
import numpy as np
import os




# adds the Krylov that wraps the AMG solver
def flopsV_manual(bare_level, levels_info, level_id, mg_solver):
    if level_id == len(levels_info)-2:
        if level_id==bare_level:
            return (2 * mg_solver.smooth_iters + 2) * levels_info[level_id].A.nnz + 0
        else:
            return (2 * mg_solver.smooth_iters + 1) * levels_info[level_id].A.nnz + 0
    else:
        if level_id==bare_level:
            return (2 * mg_solver.smooth_iters + 2) * levels_info[level_id].A.nnz + \
                   flopsV_manual(bare_level, levels_info, level_id+1, mg_solver)
        else:
            return (2 * mg_solver.smooth_iters + 1) * levels_info[level_id].A.nnz + \
                   flopsV_manual(bare_level, levels_info, level_id+1, mg_solver)




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
        trace_params['max_nr_ests'] = 100000
        trace_params['max_nr_levels'] = params['max_nr_levels']
        trace_params['problem_name'] = params['matrix_params']['problem_name']
        trace_params['nr_deflat_vctrs'] = params['nr_deflat_vctrs']
        trace_params['mlmc_deflat_vctrs'] = params['mlmc_deflat_vctrs']
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




def deflation_pre_computations(A,nr_deflat_vctrs,tolx,method,lop=None):

    if nr_deflat_vctrs>0:
        print("Computing SVD ...")
        start = time.time()

        if method=="hutchinson":
            # extract eigenpairs of Q
            print("Constructing sparse Q ...")
            Q = A.copy()
            mat_size = int(Q.shape[0]/2)
            Q[mat_size:,:] = -Q[mat_size:,:]
            print("... done")
            print("Eigendecomposing Q ...")
            Sy,Vx = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0 )
        elif method=="mlmc":
            print("Eigendecomposing Q ...")
            Sy,Vx = eigsh( lop,k=nr_deflat_vctrs,which='LM',tol=tolx )

        sgnS = np.ones(Sy.shape[0])
        for i in range(Sy.shape[0]): sgnS[i]*=(2.0*float(Sy[i]>0)-1.0)
        Sy = np.multiply(Sy,sgnS)
        Ux = np.copy(Vx)
        for idx,sgn in enumerate(sgnS) : Ux[:,idx] *= sgn
        mat_size = int(Ux.shape[0]/2)
        Ux[mat_size:,:] = -Ux[mat_size:,:]
        print("... done")
        Sx = np.diag(Sy)

        end = time.time()
        print("... done")
        print("Time to compute singular vectors (or eigenvectors) = "+str(end-start))

        try:
            nr_cores = int(os.getenv('OMP_NUM_THREADS'))
            print("IMPORTANT : this SVD decomposition was computed with "+str(nr_cores) \
                  +" cores i.e. elapsed time = "+str((end-start)*nr_cores)+" cpu seconds")
        except TypeError:
            raise Exception("Run : << export OMP_NUM_THREADS=N >>")

        start = time.time()
        # compute low-rank part of deflation
        if method=="hutchinson":
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * np.linalg.inv(Sx)
        elif method=="mlmc":
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * Sx

        tr1 = np.trace(small_A)
        end = time.time()
        print("\nTime to compute the small-matrix contribution in Deflated Hutchinson : " \
              +str(end-start))
    else:
        tr1 = 0.0
        Vx = None

    return (Vx,tr1)




# <i> is the MLMC level
def one_defl_Hutch_step(Af,Ac,mg_solver,params,method,nr_deflat_vctrs,Vx,i=0, \
                        output_params=None,P=None,R=None):

    if method=="hutchinson":

        # generate a Rademacher vector
        x = np.random.randint(2, size=Af.shape[0])
        x *= 2
        x -= 1
        x = x.astype(Af.dtype)

        if nr_deflat_vctrs>0:
            # deflating Vx from x
            x_def = x - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x))
        else:
            x_def = x

        mg_solver.level_nr = 0
        mg_solver.solve(Af,x_def,params['function_params']['tol'])
        z = mg_solver.x
        num_iters = mg_solver.num_iters

        e = np.vdot(x,z)
        itrs = num_iters

    elif method=="mlmc":

        # generate a Rademacher vector
        x0 = np.random.randint(2, size=Af.shape[0])
        x0 *= 2
        x0 -= 1
        x0 = x0.astype(Af.dtype)

        if nr_deflat_vctrs>0:
            # deflating Vx from x
            x_def = x0 - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x0))
        else:
            x_def = x0

        mg_solver.level_nr = i
        mg_solver.solve(Af,x_def,params['function_params']['tol'])
        z = mg_solver.x
        num_iters1 = mg_solver.num_iters
        output_params['results'][i]['function_iters'] += num_iters1

        xc = R*x_def

        if (i+1)==(len(mg_solver.ml.levels)-1):
            y = np.dot(mg_solver.coarsest_inv,xc)
            y = np.asarray(y).reshape(-1)
            num_iters2 = 1
        else:
            mg_solver.level_nr = i+1
            mg_solver.solve(Ac,xc,params['function_params']['tol'])
            y = mg_solver.x
            num_iters2 = mg_solver.num_iters

        output_params['results'][i+1]['function_iters'] += num_iters2

        e1 = np.vdot(x0,z)
        e2 = np.vdot(x0,P*y)
        
        e = e1-e2
        itrs = 0

    return (e,itrs)
