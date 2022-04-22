# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from math import sqrt, pow
from utils import flopsV,flopsV_manual
from aggregation import manual_aggregation
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
from numpy.linalg import eigh
from scipy.sparse import identity

from scipy.sparse.linalg import svds,eigsh,eigs
from scipy.sparse import diags

from scipy.linalg import expm
from math import exp

import time
import os

from multigrid import mg_solve
import multigrid as mg

from multigrid import MG


# ---------------------------------

class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        for idx,level in enumerate(levels):
            print("Level: "+str(idx))
            print("\tsize(R) = "+str(level.R.shape))
            print("\tsize(P) = "+str(level.P.shape))
            print("\tsize(A) = "+str(level.A.shape))

# ---------------------------------

def gamma3_application(v):
    v_size = int(v.shape[0]/2)
    v[v_size:] = -v[v_size:]
    return v

# ---------------------------------

# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, params):

    mg.level_nr = 0
    mg.coarsest_iters = 0
    mg.coarsest_iters_tot = 0
    mg.coarsest_iters_avg = 0
    mg.nr_calls = 0

    mg.coarsest_lev_iters[0] = 0

    max_nr_levels = params['max_nr_levels']

    # function params
    function_tol = params['function_params']['tol']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']

    # size of the problem
    N = A.shape[0]

    function_tol = 1e-5

    # compute the SVD (for low-rank part of deflation)
    np.random.seed(65432)
    nr_deflat_vctrs = params['nr_deflat_vctrs']

    if params['accuracy_mg_eigvs'] == 'low':
        tolx = tol=1.0e-3
        ncvx = nr_deflat_vctrs+2
    elif params['accuracy_mg_eigvs'] == 'high':
        tolx = tol=1.0e-9
        ncvx = None
    else:
        raise Exception("<accuracy_mg_eigvs> does not have a possible value.")

    # FIXME : hardcoded value for eigensolving tolerance for now
    tolx = 1.0e-14

    if nr_deflat_vctrs>0:
        print("Computing SVD (finest level) ...")
        start = time.time()
        diffA = A-A.getH()
        diffA_norm = norm( diffA,ord='fro' )
        if diffA_norm<1.0e-14:
            #Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0,ncv=ncvx )
            Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='SM',tol=tolx )
            Vx = np.copy(Ux)
        else:
            # extract eigenpairs of Q
            print("Constructing sparse Q ...")
            Q = A.copy()
            mat_size = int(Q.shape[0]/2)
            Q[mat_size:,:] = -Q[mat_size:,:]
            print("... done")
            print("Eigendecomposing Q ...")
            Sy,Vx = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0 )
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
            print("IMPORTANT : this SVD decomposition was computed with "+str(nr_cores)+" cores i.e. elapsed time = "+str((end-start)*nr_cores)+" cpu seconds")
        except TypeError:
            raise Exception("Run : << export OMP_NUM_THREADS=32 >>")

    if nr_deflat_vctrs>0:
        start = time.time()
        # compute low-rank part of deflation
        small_A = np.dot(Vx.transpose().conjugate(),Ux) * np.linalg.inv(Sx)
        tr1 = np.trace(small_A)
        end = time.time()
        print("\nTime to compute the small-matrix contribution in Deflated Hutchinson : "+str(end-start))
    else:
        tr1 = 0.0

    np.random.seed(123456)

    # pre-compute a rough estimation of the trace, to set then a tolerance
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()

    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1

        x = x.astype(A.dtype)

        mg.level_nr = 0
        z,num_iters = mg_solve( A,x,function_tol )

        e = np.vdot(x,z)
        ests[i] = e

    end = time.time()
    print("Time to compute rough estimation of trace : "+str(end-start)+"\n")

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    print("\n** rough estimation of the trace : "+str(rough_trace))
    print("")

    # then, set a rough tolerance
    rough_trace_tol = abs(trace_tol*rough_trace)

    rough_function_tol = 1e-5

    function_iters = 0
    ests = np.zeros(trace_max_nr_ests, dtype=A.dtype)

    start = time.time()
    mg.coarsest_lev_iters[0] = 0

    # main Hutchinson loop
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1

        x = x.astype(A.dtype)

        if nr_deflat_vctrs>0:
            # deflating Vx from x
            x_def = x - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x))
        else:
            x_def = x

        mg.level_nr = 0
        z,num_iters = mg_solve( A,x_def,function_tol )

        function_iters += num_iters

        e = np.vdot(x,z)

        ests[i] = e

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(   np.sum(   np.square(np.abs(ests[0:(i+1)]-ests_avg))   )/(i+1)   )
        error_est = ests_dev/sqrt(i+1)

        print(str(i)+" .. "+str(ests_avg)+" .. "+str(rough_trace)+" .. "+str(error_est)+" .. "+str(rough_trace_tol)+" .. "+str(num_iters))

        # break condition
        if i>=5 and error_est<rough_trace_tol:
            break

    end = time.time()
    print("\nTime to compute the trace with Deflated Hutchinson (excluding rough trace and excluding time for eigenvectors computation) : "+str(end-start)+"\n")

    result = dict()
    #print(tr1)
    result['trace'] = ests_avg+tr1
    result['std_dev'] = ests_dev
    result['nr_ests'] = i
    result['function_iters'] = function_iters
    result['total_complexity'] = flopsV_manual(len(mg.ml.levels), mg.ml.levels, 0)*function_iters
    result['total_complexity'] += mg.ml.levels[len(mg.ml.levels)-1].A.nnz * mg.coarsest_lev_iters[0]

    # add work due to deflation
    # FIXME : the (harcoded) factor of 3 in the following line is due to non-sparse memory accesses
    result['total_complexity'] += result['nr_ests']*(2*N*nr_deflat_vctrs)/3.0
    #print(result['nr_ests']*(2*N*nr_deflat_vctrs))

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

    # MG setup phase

    print("\nConstruction of P and A at all levels (from finest level) ...")
    start = time.time()
    aggrs = params['aggrs']
    dof = params['dof']
    mg_solver.setup(dof=dof, aggrs=aggrs, max_levels=params['max_nr_levels'], dim=2, acc_eigvs=params['accuracy_mg_eigvs'], sys_type=params['problem_name'])
    end = time.time()
    print("... done")
    
    print("Elapsed time to compute the multigrid hierarchy = "+str(end-start))
    print("IMPORTANT : this ML hierarchy was computed with 1 core i.e. elapsed time = "+str(end-start)+" cpu seconds")

    print("\nMultilevel information:")
    print(mg_solver)

    nr_levels = len(mg_solver.ml.levels)
    mg_solver.total_levels = nr_levels

    for i in range(nr_levels):
        #mg.coarsest_lev_iters[i] = 0
        mg_solver.coarsest_lev_iters[i] = 0

    if nr_levels<3:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels-1):
        mg_solver.ml.levels[i].P = csr_matrix(mg_solver.ml.levels[i].P)
        mg_solver.ml.levels[i].R = csr_matrix(mg_solver.ml.levels[i].R)

    # -----------------------------------------------------------------------------------------------

    # Rough trace estimation

    print("\nComputing rough estimation of the trace ...")

    #np.random.seed(123456)
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()

    # main Hutchinson loop
    for i in range(nr_rough_iters):
        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        x = x.astype(A.dtype)
        mg_solver.level_nr = 0
        mg_solver.solve(A,x,params['function_params']['tol'])
        z = mg_solver.x
        num_iters = mg_solver.num_iters
        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    print("... done \n")
    end = time.time()
    print("Time to compute rough estimation of the trace : "+str(end-start)+"\n")

    print("** rough value of the trace : "+str(rough_trace))

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
        tol_fraction0 = 0.5
        tol_fraction1 = 0.5
    else:
        tol_fraction0 = 0.45 #1.0/3.0
        tol_fraction1 = 0.45 #1.0/3.0

    # -----------------------------------------------------------------------------------------------

    # FIXME : these cumm things need to be removed

    cummP = sp.sparse.identity(N,dtype=A.dtype)
    cummR = sp.sparse.identity(N,dtype=A.dtype)
    cummP = csr_matrix(cummP)
    cummR = csr_matrix(cummR)

    # -----------------------------------------------------------------------------------------------

    # Compute the <difference> levels

    start = time.time()
    mg_solver.coarsest_lev_iters[0] = 0

    # coarsest-level inverse
    Acc = mg_solver.ml.levels[nr_levels-1].A
    Ncc = Acc.shape[0]
    np_Acc = Acc.todense()
    np_Acc_inv = np.linalg.inv(np_Acc)
    np_Acc_fnctn = np_Acc_inv[:,:]

    for i in range(nr_levels-1):

        if i==0 : tol_fctr = sqrt(tol_fraction0)
        elif i==1 : tol_fctr = sqrt(tol_fraction1)
        else:
            tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-3)

        level_trace_tol  = abs(params['tol']*rough_trace*tol_fctr)

        # fine and coarse matrices
        Af = mg_solver.ml.levels[i].A
        Ac =mg_solver. ml.levels[i+1].A
        # P and R
        R = mg_solver.ml.levels[i].R
        P = mg_solver.ml.levels[i].P

        print("Computing for level "+str(i)+"...")

        ests = np.zeros(params['max_nr_ests'], dtype=Af.dtype)
        for j in range(params['max_nr_ests']):

            # generate a Rademacher vector
            x0 = np.random.randint(2, size=N)
            x0 *= 2
            x0 -= 1
            x0 = x0.astype(A.dtype)
            x = cummR*x0

            mg_solver.level_nr = i
            mg_solver.solve(Af,x,params['function_params']['tol'])
            z = mg_solver.x
            num_iters = mg_solver.num_iters
            num_iters1 = num_iters
            output_params['results'][i]['function_iters'] += num_iters

            xc = R*x

            if (i+1)==(nr_levels-1):
                y = np.dot(np_Acc_fnctn,xc)
                y = np.asarray(y).reshape(-1)
                num_iters = 1
            else:
                mg_solver.level_nr = i+1
                #y,num_iters = mg_solve( Ac,xc,level_solver_tol )
                mg_solver.solve(Ac,xc,params['function_params']['tol'])
                y = mg_solver.x
                num_iters = mg_solver.num_iters

            num_iters2 = num_iters
            output_params['results'][i+1]['function_iters'] += num_iters

            e1 = np.vdot(x0,cummP*z)
            cummPh = cummP*P
            e2 = np.vdot(x0,cummPh*y)

            ests[j] = e1-e2

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            #print(str(j)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol)+" ("+str(num_iters1)+","+str(num_iters2)+")")

            # break condition
            if j>=5 and error_est<level_trace_tol:
                break

        # cummulative R and P
        cummP1 = cummP*P
        cummP = cummP1.copy()
        cummR1 = R*cummR
        cummR = cummR1.copy()

        output_params['results'][i]['nr_ests'] += j

        # set trace and standard deviation
        output_params['results'][i]['ests_avg'] = ests_avg
        output_params['results'][i]['ests_dev'] = ests_dev

        print("... done")

    # -----------------------------------------------------------------------------------------------

    # Compute now at the coarsest level

    # in case the coarsest matrix is 1x1
    if mg_solver.ml.levels[nr_levels-1].A.shape[0]==1:
        output_params['results'][nr_levels-1]['nr_ests'] += 1
        # set trace and standard deviation
        output_params['results'][nr_levels-1]['ests_avg'] = 1.0/csr_matrix(Acc)[0,0]
        output_params['results'][nr_levels-1]['ests_dev'] = 0
    else:
        if params['coarsest_level_directly']==True:
            output_params['results'][nr_levels-1]['nr_ests'] += 1
            # set trace and standard deviation
            crst_mat = cummR*cummP*np_Acc_fnctn
            output_params['results'][nr_levels-1]['ests_avg'] = np.trace(crst_mat)
            output_params['results'][nr_levels-1]['ests_dev'] = 0
        else:
            raise Exception("Stochastic coarsest-level computation is disabled at the moment.")

    # -----------------------------------------------------------------------------------------------

    end = time.time()
    print("\nTime to compute trace with MLMC (excluding rough trace and excluding setup time for the multigrid hierarchy) : "+str(end-start))

    for i in range(nr_levels-1):
        output_params['results'][i]['level_complexity'] = output_params['results'][i]['function_iters']*flopsV_manual(i, mg_solver.ml.levels, i)
        output_params['results'][i]['level_complexity'] += mg_solver.ml.levels[len(mg_solver.ml.levels)-1].A.nnz * mg_solver.coarsest_lev_iters[i]

    if params['coarsest_level_directly']==True:
        output_params['results'][nr_levels-1]['level_complexity'] = pow(mg_solver.ml.levels[nr_levels-1].A.shape[0],3) + \
                                                                    output_params['results'][nr_levels-1]['function_iters']*pow(mg_solver.ml.levels[nr_levels-1].A.shape[0],2)
    else:
        output_params['results'][nr_levels-1]['level_complexity'] = \
                     output_params['results'][nr_levels-1]['function_iters']*(mg_solver.ml.levels[nr_levels-1].A.shape[0]*mg_solver.ml.levels[nr_levels-1].A.shape[0])

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    return output_params
