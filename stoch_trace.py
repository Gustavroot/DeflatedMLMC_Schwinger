# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from math import sqrt, pow
from utils import flopsV_manual,deflation_pre_computations,one_defl_Hutch_step
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
from numpy.linalg import eigh
from scipy.sparse import identity

from scipy.sparse.linalg import svds,eigsh,eigs
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator

from scipy.linalg import expm
from math import exp

import time
import os

import matplotlib.pylab as plt
#from scipy import sparse
#import multigrid as mg

from multigrid import MG




# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, params):

    #e,V = eigsh( A,k=1,which='LM',tol=1.0e-12,sigma=0.0 )
    #print(e)
    #exit(0)

    mg_solver = MG(A)
    mg_solver.coarsest_iters = 0
    mg_solver.coarsest_iters_tot = 0
    mg_solver.coarsest_iters_avg = 0
    mg_solver.nr_calls = 0

    # size of the problem
    N = A.shape[0]

    # -----------------------------------------------------------------------------------------------

    # MG setup phase

    print("MG setup phase ...",end='',flush=True)
    start = time.time()
    aggrs = params['aggrs']
    dof = params['dof']
    mg_solver.setup(dof=dof, aggrs=aggrs, max_levels=params['max_nr_levels'], dim=2, \
                    acc_eigvs=params['accuracy_mg_eigvs'], sys_type=params['problem_name'], \
                    params=params)
    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")

    print(mg_solver)

    nr_levels = len(mg_solver.ml.levels)
    mg_solver.total_levels = nr_levels

    for i in range(nr_levels):
        mg_solver.coarsest_lev_iters[i] = 0

    if nr_levels<3:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels-1):
        mg_solver.ml.levels[i].P = csr_matrix(mg_solver.ml.levels[i].P)
        mg_solver.ml.levels[i].R = csr_matrix(mg_solver.ml.levels[i].R)

    # -----------------------------------------------------------------------------------------------

    # Pre-computations related to deflation

    print("\nResetting timer to zero ...",end='')
    mg_solver.timer.reset()
    print(" done\n")

    nr_deflat_vctrs = params['nr_deflat_vctrs']
    # tolerance of eigensolver when computing deflation vectors
    tolx = params['defl_eigvs_tol_Hutch']

    print("Computing deflation vectors ...",end='',flush=True)
    start = time.time()
    Vx,tr1 = deflation_pre_computations(A,nr_deflat_vctrs,tolx,"hutchinson",mg_solver.timer,params,mg_solver)
    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")

    print(mg_solver.timer)

    # -----------------------------------------------------------------------------------------------

    # Rough trace estimation

    print("\nComputing rough estimation of the trace ...",end='',flush=True)

    np.random.seed(123456)
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()
    # main Hutchinson loop
    for i in range(nr_rough_iters):
        #ests[i],itrs = one_defl_Hutch_step(A,None,mg_solver,params,"hutchinson",0,None,None)
        ests[i],itrs = one_defl_Hutch_step(A,None,mg_solver,params,"hutchinson",nr_deflat_vctrs,Vx,None)
    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    end = time.time()
    rough_trace += tr1
    print(" done. Time : "+str(end-start)+" seconds")
    
    # then, set a rough tolerance
    rough_trace_tol = abs(params['tol']*rough_trace)

    # -----------------------------------------------------------------------------------------------

    # Computing trace stochastically

    print("\nResetting timer to zero ...",end='')
    mg_solver.timer.reset()
    print(" done")

    print("\nComputing the trace stochastically ...",end='',flush=True)

    function_iters = 0
    ests = np.zeros(params['max_nr_ests'], dtype=A.dtype)

    start = time.time()
    mg_solver.coarsest_lev_iters[0] = 0

    # main Hutchinson loop
    for i in range(params['max_nr_ests']):

        ests[i],itrs = one_defl_Hutch_step(A,None,mg_solver,params,"hutchinson",nr_deflat_vctrs,Vx,None)
        function_iters += itrs

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(   np.sum(   np.square(np.abs(ests[0:(i+1)]-ests_avg))   )/(i+1)   )
        
        error_est = ests_dev/sqrt(i+1)

        # break condition
        print(ests_dev)
        print(error_est)
        print(rough_trace_tol)
        if i>=5 and error_est<rough_trace_tol:
            break

    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")
    
    #print("\nTime to compute the trace with Deflated Hutchinson (excluding rough trace and excluding time for eigenvectors computation) : "+str(end-start)+"\n")

    # -----------------------------------------------------------------------------------------------

    # Gathering results

    result = dict()
    result['trace'] = ests_avg+tr1
    result['std_dev'] = ests_dev
    result['nr_ests'] = i
    result['function_iters'] = function_iters
    result['total_complexity'] = flopsV_manual(len(mg_solver.ml.levels), mg_solver.ml.levels, 0, mg_solver)*function_iters
    result['total_complexity'] += mg_solver.ml.levels[len(mg_solver.ml.levels)-1].A.nnz * mg_solver.coarsest_lev_iters[0]

    # add work due to deflation
    # FIXME : the (harcoded) factor of 3 in the following line is due to non-sparse memory accesses
    result['total_complexity'] += result['nr_ests']*(2*N*nr_deflat_vctrs)/3.0

    print(mg_solver.timer)

    return result




# compute tr(A^{-1}) via MLMC
def mlmc(A, params):

    mg_solver = MG(A)
    mg_solver.coarsest_iters = 0
    mg_solver.coarsest_iters_tot = 0
    mg_solver.coarsest_iters_avg = 0
    mg_solver.nr_calls = 0

    # size of the problem
    N = A.shape[0]

    # -----------------------------------------------------------------------------------------------

    # skipping second level only, for now
    if len(params['mlmc_levels_to_skip'])>1:
        raise Exception("Only allowed to skip one level for now")
    if len(params['mlmc_levels_to_skip'])==1:
        skip_level = True
    else:
        skip_level = False
    if skip_level==True and not params['mlmc_levels_to_skip'][0]==1:
        raise Exception("Only allowed to skip the second level for now")

    # -----------------------------------------------------------------------------------------------

    # MG setup phase

    print("MG setup phase ...",end='',flush=True)
    start = time.time()
    mg_solver.setup(dof=params['dof'], aggrs=params['aggrs'], max_levels=params['max_nr_levels'], dim=2, \
                    acc_eigvs=params['accuracy_mg_eigvs'], sys_type=params['problem_name'], \
                    params=params)
    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")
    
    print(mg_solver)

    nr_levels = len(mg_solver.ml.levels)
    mg_solver.total_levels = nr_levels

    if nr_levels<3:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels):
        mg_solver.coarsest_lev_iters[i] = 0

    for i in range(nr_levels-1):
        mg_solver.ml.levels[i].P = csr_matrix(mg_solver.ml.levels[i].P)
        mg_solver.ml.levels[i].R = csr_matrix(mg_solver.ml.levels[i].R)

    # this is important to compute the deflation vectors for the <difference> operators
    mg_solver.skip_level = skip_level

    # -----------------------------------------------------------------------------------------------

    # Pre-computations related to deflation

    print("\nResetting timer to zero ...",end='')
    mg_solver.timer.reset()
    print(" done\n")

    print("Computing deflation vectors ...",end='',flush=True)
    start = time.time()
    # this parameter tells us how many times less deflation vectors we need in MLMC
    nr_deflat_vctrs = params['mlmc_deflat_vctrs']
    # tolerance of eigensolver when computing deflation vectors
    tolx = params['defl_eigvs_tol_MLMC']

    Vxs = []
    Uxs = []
    tr1s = []

    for ix in range(nr_levels-1):
    
        if skip_level and ix==1:
            Vxs.append([])
            Uxs.append([])
            tr1s.append(0.0)
            continue

        mg_solver.level_for_diff_op = ix
        lop = LinearOperator(mg_solver.ml.levels[ix].A.shape, matvec=mg_solver.diff_op_Q)
        Vx,Ux,tr1 = deflation_pre_computations(A,nr_deflat_vctrs[ix],tolx,"mlmc",mg_solver.timer,params,mg_solver,lop,level_nr=ix)
        Vxs.append(Vx)
        Uxs.append(Ux)
        tr1s.append(tr1)
    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")

    print(mg_solver.timer)

    # -----------------------------------------------------------------------------------------------

    # Rough trace estimation

    # tolerance of eigensolver when computing deflation vectors
    print("Computing deflation vectors (for rough trace estimation purposes only) ...",end='',flush=True)
    start = time.time()
    Vx,tr1 = deflation_pre_computations(A,params['nr_deflat_vctrs'],params['defl_eigvs_tol_Hutch'],"hutchinson", \
                                        mg_solver.timer,params,mg_solver)
    end = time.time()
    print(" done. Time : "+str(end-start)+" seconds")

    np.random.seed(123456)

    print("\nComputing rough estimation of the trace ...",end='',flush=True)
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()
    # main Hutchinson loop
    for i in range(nr_rough_iters):
        #ests[i],itrs = one_defl_Hutch_step(A,None,mg_solver,params,"hutchinson",0,None,None)
        ests[i],itrs = one_defl_Hutch_step(A,None,mg_solver,params,"hutchinson",params['nr_deflat_vctrs'],Vx,None)
    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    end = time.time()
    rough_trace += tr1
    print(" done. Time : "+str(end-start)+" seconds")

    # -----------------------------------------------------------------------------------------------

    # Zeroing counters

    # setting to zero the counters and results to be returned
    output_params = dict()
    output_params['nr_levels'] = nr_levels
    output_params['trace'] = 0.0
    output_params['total_complexity'] = 0.0
    output_params['std_dev'] = 0.0
    output_params['results'] = list()
    for i in range(nr_levels):
        output_params['results'].append(dict())
        output_params['results'][i]['function_iters'] = 0
        output_params['results'][i]['nr_ests'] = 0
        output_params['results'][i]['ests_avg'] = 0.0
        output_params['results'][i]['ests_dev'] = 0.0
        output_params['results'][i]['level_complexity'] = 0.0

    # -----------------------------------------------------------------------------------------------

    # delta factors for MLMC

    if nr_levels<3 : raise Exception("Number of levels restricted to >2 for now ...")
    if nr_levels==3:
        tol_fraction0 = 0.8
        tol_fraction1 = 0.2
    else:
        tol_fraction0 = 0.45 #1.0/3.0
        tol_fraction1 = 0.45 #1.0/3.0

    if skip_level:
        tol_fraction0 = tol_fraction0 + tol_fraction1

    # -----------------------------------------------------------------------------------------------

    # Compute the <difference> levels

    print("\nResetting timer to zero ...",end='')
    mg_solver.timer.reset()
    print(" done\n")

    mg_solver.coarsest_lev_iters[0] = 0

    for i in range(nr_levels-1):

        if skip_level and i==1:
            continue

        start = time.time()

        # setting elta factor at level i
        if i==0 : tol_fctr = sqrt(tol_fraction0)
        elif i==1 : tol_fctr = sqrt(tol_fraction1)
        else:
            if skip_level:
                tol_fctr = sqrt(1.0-tol_fraction0)/sqrt(nr_levels-3)
            else:
                tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-3)

        level_trace_tol  = abs(params['tol']*rough_trace*tol_fctr)

        if skip_level and i==0:
            # fine and coarse matrices
            Af = mg_solver.ml.levels[i].A
            Ac = mg_solver.ml.levels[i+1+1].A
            # P and R
            R0 = mg_solver.ml.levels[i].R
            P0 = mg_solver.ml.levels[i].P
            R1 = mg_solver.ml.levels[i+1].R
            P1 = mg_solver.ml.levels[i+1].P
        else:
            # fine and coarse matrices
            Af = mg_solver.ml.levels[i].A
            Ac = mg_solver.ml.levels[i+1].A
            # P and R
            R = mg_solver.ml.levels[i].R
            P = mg_solver.ml.levels[i].P

        print("Computing for level "+str(i)+" ...",end='',flush=True)

        ests = np.zeros(params['max_nr_ests'], dtype=Af.dtype)
        for j in range(params['max_nr_ests']):

            if skip_level and i==0:
                ests[j],itrs = one_defl_Hutch_step(Af,Ac,mg_solver,params,"mlmc",nr_deflat_vctrs[i],Vxs[i],Uxs[i],i,output_params,P0,R0,P1,R1)
            else:
                ests[j],itrs = one_defl_Hutch_step(Af,Ac,mg_solver,params,"mlmc",nr_deflat_vctrs[i],Vxs[i],Uxs[i],i,output_params,P,R)

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))

            error_est = ests_dev/sqrt(j+1)

            #print(ests_dev)
            #print(error_est)
            #print(level_trace_tol)

            # break condition
            if j>=5 and error_est<level_trace_tol:
                break

        output_params['results'][i]['nr_ests'] += j

        # set trace and standard deviation
        output_params['results'][i]['ests_avg'] = ests_avg+tr1s[i]
        output_params['results'][i]['ests_dev'] = ests_dev

        end = time.time()

        print(" done. Time : "+str(end-start)+" seconds")

    # Compute now at the coarsest level

    # in case the coarsest matrix is 1x1
    if mg_solver.ml.levels[nr_levels-1].A.shape[0]==1:
        raise Exception("your coarsest-level matrix is of size 1 ... is this what you want?")
        output_params['results'][nr_levels-1]['nr_ests'] += 1
        # set trace and standard deviation
        output_params['results'][nr_levels-1]['ests_avg'] = 1.0/csr_matrix(Acc)[0,0]
        output_params['results'][nr_levels-1]['ests_dev'] = 0
    else:
        if params['coarsest_level_directly']==True:
            output_params['results'][nr_levels-1]['nr_ests'] += 1
            # set trace and standard deviation
            crst_mat = mg_solver.coarsest_inv
            if params["use_permuted"]:
                crst_mat = mg_solver.ml.levels[nr_levels-1].Pperm.transpose().conjugate() * (crst_mat * mg_solver.ml.levels[nr_levels-1].Bblock_perm)
            output_params['results'][nr_levels-1]['ests_avg'] = np.trace(crst_mat)
            output_params['results'][nr_levels-1]['ests_dev'] = 0
        else:
            raise Exception("Stochastic coarsest-level computation is disabled at the moment.")

    # -----------------------------------------------------------------------------------------------

    # Gathering results

    for i in range(nr_levels-1):
        output_params['results'][i]['level_complexity'] = \
            output_params['results'][i]['function_iters']*flopsV_manual(i, mg_solver.ml.levels, \
            i, mg_solver)
        output_params['results'][i]['level_complexity'] += \
            mg_solver.ml.levels[len(mg_solver.ml.levels)-1].A.nnz * \
            mg_solver.coarsest_lev_iters[i]

    if params['coarsest_level_directly']==True:
        output_params['results'][nr_levels-1]['level_complexity'] = \
            pow(mg_solver.ml.levels[nr_levels-1].A.shape[0],3) + \
            output_params['results'][nr_levels-1]['function_iters']* \
            pow(mg_solver.ml.levels[nr_levels-1].A.shape[0],2)
    else:
        output_params['results'][nr_levels-1]['level_complexity'] = \
                     output_params['results'][nr_levels-1]['function_iters']*\
                     (mg_solver.ml.levels[nr_levels-1].A.shape[0]* \
                     mg_solver.ml.levels[nr_levels-1].A.shape[0])

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print(mg_solver.timer)

    return output_params
