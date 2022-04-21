# Stochastic methods for the computation of the trace of the inverse

import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
from utils import flopsV,flopsV_manual
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
#import png
from numpy.linalg import eigh
from scipy.sparse import identity

from scipy.sparse.linalg import svds,eigsh,eigs
from scipy.sparse import diags

from scipy.linalg import expm
from math import exp

import time
import os

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection,symmetric_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation

from multigrid import mg_solve
import multigrid as mg


# ---------------------------------

# specific to LQCD

class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        return "For manual aggregation, printing <ml> is under construction"

# ---------------------------------

def gamma3_application(v):
    v_size = int(v.shape[0]/2)
    v[v_size:] = -v[v_size:]
    return v

def gamma5_application(v,l,dof):

    sz = v.shape[0]
    for i in range(int(sz/dof[l])):
        # negate first half
        for j in range(int(dof[l]/2)):
            v[i*dof[l]+j] = -v[i*dof[l]+j]

    return v

"""
# https://stackoverflow.com/questions/33713221/create-png-image-from-sparse-data
def write_png(A, filename):
    m, n = A.shape

    w = png.Writer(n, m, greyscale=True, bitdepth=1)

    class RowIterator:
        def __init__(self, A):
            self.A = A.tocsr()
            self.current = 0
            return

        def __iter__(self):
            return self

        def __next__(self):
            if self.current+1 > A.shape[0]:
                raise StopIteration
            out = np.ones(A.shape[1], dtype=bool)
            out[self.A[self.current].indices] = False
            self.current += 1
            return out

    with open(filename, 'wb') as f:
        w.write(f, RowIterator(A))

    return
"""


# ---------------------------------

# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, function, params):

    mg.level_nr = 0
    mg.coarsest_iters = 0
    mg.coarsest_iters_tot = 0
    mg.coarsest_iters_avg = 0
    mg.nr_calls = 0

    mg.coarsest_lev_iters[0] = 0

    #if params['problem_name']=='schwinger':
    #    global ml

    # TODO : check input params !

    max_nr_levels = params['max_nr_levels']

    # function params
    function_name = params['function_params']['function_name']
    spec_name = params['function_params']['spec_name']
    function_tol = params['function_params']['tol']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']

    use_Q = params['use_Q']

    # size of the problem
    N = A.shape[0]

    function_tol = 1e-5

    if use_Q:
        print("Constructing sparse Q ...")
        Q = A.copy()
        mat_size = int(Q.shape[0]/2)
        Q[mat_size:,:] = -Q[mat_size:,:]
        print("... done")

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
        if use_Q:
            #Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0,ncv=ncvx )

            #Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0 )

            Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='SM',tol=tolx )

            Vx = np.copy(Ux)
        else:
            diffA = A-A.getH()
            diffA_norm = norm( diffA,ord='fro' )
            if diffA_norm<1.0e-14:
                #Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0,ncv=ncvx )
                Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='SM',tol=tolx )
                Vx = np.copy(Ux)
            else:
                if params['problem_name']=='schwinger':
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
                else:
                    Ux,Sy,Vy = svds( A,k=nr_deflat_vctrs,which='SM',tol=tolx )
                    Vx = Vy.transpose().conjugate()
        Sx = np.diag(Sy)
        end = time.time()
        print("... done")
        print("Time to compute singular vectors (or eigenvectors) = "+str(end-start))

        try:
            nr_cores = int(os.getenv('OMP_NUM_THREADS'))
            print("IMPORTANT : this SVD decomposition was computed with "+str(nr_cores)+" cores i.e. elapsed time = "+str((end-start)*nr_cores)+" cpu seconds")
        except TypeError:
            raise Exception("Run : << export OMP_NUM_THREADS=32 >>")

    # tr( A ) = tr(U1H*U1* S1 ) + tr( A *(I-U1*U1H))
    # tr( exp(-L) ) = tr(U1H*U1* exp(-S1) ) + tr( exp(-L) *(I-U1*U1H))
    # tr( exp(-L) *(I-U1*U1H)) = (1/n)sum^{n} ( ziH * EXPOFIT( -L,(I-U1*U1H)*zi ) )

    if nr_deflat_vctrs>0:
        start = time.time()
        # compute low-rank part of deflation
        if not function_name=="exponential":
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * np.linalg.inv(Sx)
        else:
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * expm(-Sx)
        tr1 = np.trace(small_A)
        end = time.time()
        print("\nTime to compute the small-matrix contribution in Deflated Hutchinson : "+str(end-start))
    else:
        tr1 = 0.0

    if function_name=="exponential":
        #print("Sending A as dense ...")
        function.putvalue('Ax',A.todense())
        #print("... done")

    """
    # TODO : remove this section
    print("\nComputing theoretical trace ...")
    if function_name=="exponential":
        Ax = -A
        trx = np.trace( expm(Ax.todense()) )
    else:
        trx = np.trace(np.linalg.inv( A.todense() ))
    print("tr(f(A)) = "+str(trx))
    print("... done")
    """

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
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        if params['problem_name']=='schwinger':
            mg.level_nr = 0
            z,num_iters = mg_solve( A,x,function_tol )
        else:
            z,num_iters = function_sparse(A,x,function_tol,function,function_name,spec_name)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        e = np.vdot(x,z)
        ests[i] = e

    end = time.time()
    print("Time to compute rough estimation of trace : "+str(end-start)+"\n")

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    print("\n** rough estimation of the trace : "+str(rough_trace))
    print("")

    # then, set a rough tolerance
    rough_trace_tol = abs(trace_tol*rough_trace)
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    #rough_solver_tol = abs(rough_trace_tol/N)

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
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

        x = x.astype(A.dtype)

        if nr_deflat_vctrs>0:
            # deflating Vx from x
            x_def = x - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x))
        else:
            x_def = x

        if use_Q:
            if params['problem_name']=='schwinger':
                x_def = gamma3_application(x_def)
            elif params['problem_name']=='LQCD':
                x_def = gamma5_application(x_def,0,params['dof'])

        if params['problem_name']=='schwinger':
            mg.level_nr = 0
            z,num_iters = mg_solve( A,x_def,function_tol )
        else:
            z,num_iters = function_sparse(A,x_def,rough_function_tol,function,function_name,spec_name)

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
    if function_name=="inverse" and  spec_name=='mg':

        if not params['problem_name']=='schwinger':
            result['total_complexity'] = flopsV(len(function.levels), function.levels, 0)*function_iters
        else:
            result['total_complexity'] = flopsV_manual(len(mg.ml.levels), mg.ml.levels, 0)*function_iters
            result['total_complexity'] += mg.ml.levels[len(mg.ml.levels)-1].A.nnz * mg.coarsest_lev_iters[0]

        # add work due to deflation
        # FIXME : the (harcoded) factor of 3 in the following line is due to non-sparse memory accesses
        result['total_complexity'] += result['nr_ests']*(2*N*nr_deflat_vctrs)/3.0
        #print(result['nr_ests']*(2*N*nr_deflat_vctrs))

    return result


# compute tr(A^{-1}) via MLMC
def mlmc(A, function, params):

    #if params['problem_name']=='schwinger':
    #    global ml

    # TODO : check input params !

    # function params
    function_name = params['function_params']['function_name']
    spec_name = params['function_params']['spec_name']
    function_tol = params['function_params']['tol']
    lambda_min = params['function_params']['lambda_min']

    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']
    trace_ml_constr = params['multilevel_construction']

    use_Q = params['use_Q']

    # size of the problem
    N = A.shape[0]

    max_nr_levels = params['max_nr_levels']

    print("\nConstruction of P and A at all levels (from finest level) ...")
    start = time.time()
    if trace_ml_constr=='pyamg':
        if params['aggregation_type']=='SA':
            mg.ml = pyamg.smoothed_aggregation_solver( A,max_levels=max_nr_levels )
        elif params['aggregation_type']=='ASA':
            [mg.ml, work] = adaptive_sa_solver(A, num_candidates=1, candidate_iters=2, improvement_iters=8,
                                            strength='symmetric', aggregate='standard', max_levels=max_nr_levels)
        else:
            raise Exception("Aggregation type not specified for PyAMG")
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)

    elif trace_ml_constr=='direct_interpolation':

        mg.ml = SimpleML()
        # appending level 0
        mg.ml.levels.append(LevelML())
        mg.ml.levels[0].A = A.copy()

        for i in range(max_nr_levels-1):

            #print(i)

            print(mg.ml.levels[i].A)

            S = symmetric_strength_of_connection(mg.ml.levels[i].A)
            splitting = split.RS(S)

            mg.ml.levels[i].P = 1.0 * direct_interpolation(mg.ml.levels[i].A, S, splitting)
            mg.ml.levels[i].R = 1.0 * mg.ml.levels[i].P.transpose().conjugate().copy()

            print(mg.ml.levels[i].P)
            print(mg.ml.levels[i].P.count_nonzero())
            print(mg.ml.levels[i].P * mg.ml.levels[i].P.transpose().conjugate())

            mg.ml.levels.append(LevelML())
            mg.ml.levels[i+1].A = csr_matrix( mg.ml.levels[i].R * mg.ml.levels[i].A * mg.ml.levels[i].P )

    # specific to Schwinger
    elif trace_ml_constr=='manual_aggregation':

        # TODO : get <aggr_size> from input params
        # TODO : get <dof_size> from input params

        aggrs = params['aggrs']
        dof = params['dof']

        # 128 ---> 64 ---> 8 ---> 2

        mg.ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2, acc_eigvs=params['accuracy_mg_eigvs'], sys_type=params['problem_name'])

    # specific to LQCD
    elif trace_ml_constr=='from_files':

        import scipy.io as sio

        mg.ml = SimpleML()

        # load A at each level
        for i in range(max_nr_levels):
            mg.ml.levels.append(LevelML())
            if i==0:
                mat_contents = sio.loadmat('LQCD_A'+str(i+1)+'.mat')
                Axx = mat_contents['A'+str(i+1)]
                mg.ml.levels[i].A = Axx.copy()

        # load P at each level
        for i in range(max_nr_levels-1):
            mat_contents = sio.loadmat('LQCD_P'+str(i+1)+'.mat')
            Pxx = mat_contents['P'+str(i+1)]
            mg.ml.levels[i].P = Pxx.copy()
            # construct R from P
            Rxx = Pxx.copy()
            Rxx = Rxx.conjugate()
            Rxx = Rxx.transpose()
            mg.ml.levels[i].R = Rxx.copy()

        # build the other A's
        for i in range(1,max_nr_levels):
            mg.ml.levels[i].A = mg.ml.levels[i-1].R*mg.ml.levels[i-1].A*mg.ml.levels[i-1].P

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")
    end = time.time()
    print("... done")
    print("Elapsed time to compute the multigrid hierarchy = "+str(end-start))
    print("IMPORTANT : this ML hierarchy was computed with 1 core i.e. elapsed time = "+str(end-start)+" cpu seconds")

    print("\nMultilevel information:")
    print(mg.ml)

    # the actual number of levels
    nr_levels = len(mg.ml.levels)
    mg.total_levels = nr_levels

    for i in range(nr_levels):
        mg.coarsest_lev_iters[i] = 0

    if nr_levels<3:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels):
        print("size(A"+str(i)+") = "+str(mg.ml.levels[i].A.shape[0])+"x"+str(mg.ml.levels[i].A.shape[1]))
    for i in range(nr_levels-1):
        print("size(P"+str(i)+") = "+str(mg.ml.levels[i].P.shape[0])+"x"+str(mg.ml.levels[i].P.shape[1]))

    print("")

    #exit(0)

    #for i in range(nr_levels-1):
    #    ml.levels[i].P = ml.levels[i].P.astype(A.dtype)
    #    ml.levels[i].R = ml.levels[i].R.astype(A.dtype)

    for i in range(nr_levels-1):
        mg.ml.levels[i].P = csr_matrix(mg.ml.levels[i].P)
        mg.ml.levels[i].R = csr_matrix(mg.ml.levels[i].R)

    """
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    Bx1 = ml.levels[0].A.copy()
    Bx2 = ml.levels[1].A.copy()
    Bx3 = ml.levels[2].A.copy()

    Cx1 = ml.levels[0].P.copy()
    Dx1 = ml.levels[0].R.copy()
    Cx2 = ml.levels[1].P.copy()
    Dx2 = ml.levels[1].R.copy()

    if not (function_name=="exponential"):
        Bx1inv = np.linalg.inv(Bx1.todense())
        Bx2inv = np.linalg.inv(Bx2.todense())
        Bx3inv = np.linalg.inv(Bx3.todense())
    else:
        Bx1inv = expm(-Bx1).todense()
        Bx2inv = expm(-Bx2).todense()
        Bx3inv = expm(-Bx3).todense()

    Bx2invproj = Cx1*Bx2inv*Dx1
    Bx3invproj = (Cx1*Cx2)*Bx3inv*(Dx2*Dx1)

    diffxinv1 = Bx1inv - Bx2invproj
    diffxinv1xx = diffxinv1 + diffxinv1.transpose()

    diffxinv2 = Bx2invproj - Bx3invproj
    diffxinv2xx = diffxinv2 + diffxinv2.transpose()

    offdiag_fro_norm1 = npnorm( diffxinv1xx-np.diag(np.diag(diffxinv1xx)), ord='fro' )
    offdiag_fro_norm2 = npnorm( diffxinv2xx-np.diag(np.diag(diffxinv2xx)), ord='fro' )

    print("\nTheoretical estimation of variance, diff at level 1 : "+str(0.5*offdiag_fro_norm1*offdiag_fro_norm1))
    print("Theoretical estimation of variance, diff at level 2 : "+str(0.5*offdiag_fro_norm2*offdiag_fro_norm2))

    # -------------

    pure_x = Bx1inv+Bx1inv.transpose()
    offdiag_fro_norm = npnorm( pure_x-np.diag(np.diag(pure_x)), ord='fro' )
    print("Theoretical estimation of variance, pure : "+str(0.5*offdiag_fro_norm*offdiag_fro_norm))
    print("")

    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    """

    # FIXME : theoretical esimation goes here

    if not (function_name=="exponential"):
        if not params['problem_name']=='schwinger':
            print("\nCreating solver with PyAMG for each level ...")
            ml_solvers = list()
            for i in range(nr_levels-1):
                [mlx, work] = adaptive_sa_solver(mg.ml.levels[i].A, num_candidates=2, candidate_iters=5, improvement_iters=8,
                                                 strength='symmetric', aggregate='standard', max_levels=9)
                ml_solvers.append(mlx)
            print("... done")

    function_tol = 1e-5

    if function_name=="exponential":
        #print("Sending A as dense ...")
        function.putvalue('Ax',A.todense())
        #print("... done")

    """
    # TODO : remove this section
    print("\nComputing theoretical trace ...")
    if function_name=="exponential":
        Ax = -A
        trx = np.trace( expm(Ax.todense()) )
    else:
        trx = np.trace(np.linalg.inv( A.todense() ))
    print("tr(f(A)) = "+str(trx))
    print("... done")
    """

    print("\nComputing rough estimation of the trace ...")

    np.random.seed(123456)

    # pre-compute a rough estimate of the trace, to set then a tolerance
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()

    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        #x = np.random.randint(4, size=N)
        #x *= 2
        #x -= 1
        #x = np.where(x==3, -1j, x)
        #x = np.where(x==5, 1j, x)

        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        if not (function_name=="exponential"):
            if params['problem_name']=='schwinger':
                mg.level_nr = 0
                z,num_iters = mg_solve( A,x,function_tol )
            else:
            
                if spec_name=='cg' or spec_name=='gmres':
                    z,num_iters = function_sparse(A,x,function_tol,function,function_name,spec_name)
                elif spec_name=='mg':
                    z,num_iters = function_sparse(A,x,function_tol,ml_solvers[0],function_name,spec_name)
                else:
                    raise Exception('Only cg, gmres or mg allowed for solvers at the moment')
        else:
            z,num_iters = function_sparse(A,x,function_tol,function,function_name,spec_name)

        #print(num_iters)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        e = np.vdot(x,z)
        ests[i] = e

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)

    print("... done \n")

    end = time.time()
    print("Time to compute rough estimation of the trace : "+str(end-start)+"\n")

    print("** rough estimation of the trace : "+str(rough_trace))

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

    #return output_params

    # compute level differences
    #level_trace_tol  = abs(trace_tol*rough_trace/sqrt(nr_levels-1))
    #level_solver_tol = level_trace_tol/N
    #level_solver_tol = 1e-9/sqrt(nr_levels)
    level_solver_tol = 1e-5

    print("")

    # FIXME : this has to be generalized slightly to include stochastic coarsest level
    if nr_levels<3 : raise Exception("Number of levels restricted to >2 for now ...")
    if nr_levels==3:
        tol_fraction0 = 0.5
        tol_fraction1 = 0.5
    else:
        tol_fraction0 = 0.45 #1.0/3.0
        tol_fraction1 = 0.45 #1.0/3.0

    cummP = sp.sparse.identity(N,dtype=A.dtype)
    cummR = sp.sparse.identity(N,dtype=A.dtype)
    cummP = csr_matrix(cummP)
    cummR = csr_matrix(cummR)

    start = time.time()
    mg.coarsest_lev_iters[0] = 0

    if not (function_name=="exponential"):
        # coarsest-level inverse
        Acc = mg.ml.levels[nr_levels-1].A
        Ncc = Acc.shape[0]
        np_Acc = Acc.todense()
        np_Acc_inv = np.linalg.inv(np_Acc)
        np_Acc_fnctn = np_Acc_inv[:,:]
    else:
        np_Acc_fnctn = expm( -mg.ml.levels[nr_levels-1].A )

    for i in range(nr_levels-1):

        if i==0 : tol_fctr = sqrt(tol_fraction0)
        elif i==1 : tol_fctr = sqrt(tol_fraction1)
        # e.g. sqrt(0.45), sqrt(0.45), sqrt(0.1*(1.0/(nl-2)))
        else :
            if params['coarsest_level_directly']==True:
                tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-3)
            else:
                tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-2)

        level_trace_tol  = abs(trace_tol*rough_trace*tol_fctr)

        # fine and coarse matrices
        Af = mg.ml.levels[i].A
        Ac =mg. ml.levels[i+1].A
        # P and R
        R = mg.ml.levels[i].R
        P = mg.ml.levels[i].P

        print("Computing for level "+str(i)+"...")

        ests = np.zeros(trace_max_nr_ests, dtype=Af.dtype)
        for j in range(trace_max_nr_ests):

            # generate a Rademacher vector
            x0 = np.random.randint(2, size=N)
            x0 *= 2
            x0 -= 1
            #x = np.random.randint(4, size=N)
            #x *= 2
            #x -= 1
            #x = np.where(x==3, -1j, x)
            #x = np.where(x==5, 1j, x)

            x0 = x0.astype(A.dtype)

            x = cummR*x0

            if use_Q:
                if params['problem_name']=='schwinger':
                    x = gamma3_application(x)
                elif params['problem_name']=='LQCD':
                    x = gamma5_application(x,i,params['dof'])

            if function_name=="exponential":
                function.putvalue('Ax',Af.todense())

            if not (function_name=="exponential"):
                if params['problem_name']=='schwinger':
                    mg.level_nr = i
                    z,num_iters = mg_solve( Af,x,level_solver_tol )
                    #print(num_iters)
                else:

                    if spec_name=='cg' or spec_name=='gmres':
                        z,num_iters = function_sparse(Af,x,level_solver_tol,function,function_name,spec_name)
                    elif spec_name=='mg':
                        z,num_iters = function_sparse(Af,x,level_solver_tol,ml_solvers[i],function_name,spec_name)
                    else:
                        raise Exception('Only cg, gmres or mg allowed for solvers at the moment')
            else:
                z,num_iters = function_sparse(Af,x,level_solver_tol,function,function_name,spec_name)

            num_iters1 = num_iters

            output_params['results'][i]['function_iters'] += num_iters

            # reverting back the application of gamma
            if use_Q:
                if params['problem_name']=='schwinger':
                    x = gamma3_application(x)
                elif params['problem_name']=='LQCD':
                    x = gamma5_application(x,i,params['dof'])

            xc = R*x

            if use_Q:
                if params['problem_name']=='schwinger':
                    xc = gamma3_application(xc)
                elif params['problem_name']=='LQCD':
                    xc = gamma5_application(xc,i+1,params['dof'])

            if (i+1)==(nr_levels-1):
                # apply function <directly> i.e. not iteratively
                #np_Ac = Ac.todense()
                if function_name=="exponential":
                    np_Acx = np_Acc_fnctn.todense()
                    y = np.dot(np_Acx,xc)
                else:
                    y = np.dot(np_Acc_fnctn,xc)
                y = np.asarray(y).reshape(-1)
                num_iters = 1
            else:
                if function_name=="exponential":
                    function.putvalue('Ax',Ac.todense())

                if not (function_name=="exponential"):
                    if params['problem_name']=='schwinger':
                        mg.level_nr = i+1
                        y,num_iters = mg_solve( Ac,xc,level_solver_tol )
                        #print(num_iters)
                    else:

                        if spec_name=='cg' or spec_name=='gmres':
                            y,num_iters = function_sparse(Ac,xc,level_solver_tol,function,function_name,spec_name)
                        elif spec_name=='mg':
                            y,num_iters = function_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],function_name,spec_name)
                        else:
                            raise Exception('Only cg, gmres or mg allowed for solvers at the moment')
                else:
                    y,num_iters = function_sparse(Ac,xc,level_solver_tol,function,function_name,spec_name)
            #y,num_iters = solver_sparse(Ac,xc,level_solver_tol,ml_solvers[i+1],"mg")

            num_iters2 = num_iters

            output_params['results'][i+1]['function_iters'] += num_iters

            e1 = np.vdot(x0,cummP*z)
            cummPh = cummP*P
            #yrh = cummPh*y
            e2 = np.vdot(x0,cummPh*y)

            ests[j] = e1-e2

            # average of estimates
            ests_avg = np.sum(ests[0:(j+1)])/(j+1)
            # and standard deviation
            ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(j+1)]-ests_avg)))/(j+1))
            error_est = ests_dev/sqrt(j+1)

            print(str(j)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol)+" ("+str(num_iters1)+","+str(num_iters2)+")")

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

    # compute now at the coarsest level

    # in case the coarsest matrix is 1x1
    if mg.ml.levels[nr_levels-1].A.shape[0]==1:

        output_params['results'][nr_levels-1]['nr_ests'] += 1
        # set trace and standard deviation
        if not (function_name=="exponential"):
            output_params['results'][nr_levels-1]['ests_avg'] = 1.0/csr_matrix(Acc)[0,0]
        else:
            output_params['results'][nr_levels-1]['ests_avg'] = exp(csr_matrix(Acc)[0,0])
        output_params['results'][nr_levels-1]['ests_dev'] = 0

    else:

        if params['coarsest_level_directly']==True:

            output_params['results'][nr_levels-1]['nr_ests'] += 1
            # set trace and standard deviation

            if use_Q:
                print("Constructing sparse Q ...")
                Qcc = Acc.copy()
                mat_size = int(Qcc.shape[0]/2)
                Qcc[mat_size:,:] = -Qcc[mat_size:,:]
                print("... done")
                output_params['results'][nr_levels-1]['ests_avg'] = np.trace(np.linalg.inv( Qcc.todense() ))
            else:
                crst_mat = cummR*cummP*np_Acc_fnctn
                if function_name=="exponential":
                    output_params['results'][nr_levels-1]['ests_avg'] = np.trace(crst_mat.todense())
                else:
                    output_params['results'][nr_levels-1]['ests_avg'] = np.trace(crst_mat)

            output_params['results'][nr_levels-1]['ests_dev'] = 0

        else:

            ests = np.zeros(trace_max_nr_ests, dtype=Acc.dtype)

            tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-2)
            level_trace_tol  = abs(trace_tol*rough_trace*tol_fctr)

            # ----- extracting eigenvectors for deflation

            if use_Q:
                print("Constructing sparse Q ...")
                Qcc = Acc.copy()
                mat_size = int(Qcc.shape[0]/2)
                Qcc[mat_size:,:] = -Qcc[mat_size:,:]
                print("... done")

            # compute the SVD (for low-rank part of deflation)
            np.random.seed(65432)
            # more deflation vectors for the coarsest level
            nr_deflat_vctrs = 16

            if nr_deflat_vctrs>(Acc.shape[0]-2) : nr_deflat_vctrs=Acc.shape[0]-2
            print("Computing SVD (coarsest level) ...")
            if use_Q:
                Sy,Ux = eigsh( Qcc,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
                Vx = np.copy(Ux)
            else:
                diffA = Acc-Acc.getH()
                diffA_norm = norm( diffA,ord='fro' )
                if diffA_norm<1.0e-14:
                    Sy,Ux = eigsh( Acc,k=nr_deflat_vctrs,which='LM',tol=1.0e-5,sigma=0.0 )
                    Vx = np.copy(Ux)
                else:
                    Ux,Sy,Vy = svds(Acc, k=nr_deflat_vctrs, which='SM', tol=1.0e-5)
                    Vx = Vy.transpose().conjugate()
            Sx = np.diag(Sy)
            print("... done")

            # compute low-rank part of deflation
            small_A1 = cummP*Ux
            small_A2 = cummR*small_A1
            small_A3 = np.dot(Vx.transpose().conjugate(),small_A2)
            small_A = small_A3*np.linalg.inv(Sx)
            tr1c = np.trace(small_A)

            # -------------------------

            for i in range(trace_max_nr_ests):

                # generate a Rademacher vector

                xc = np.random.randint(2, size=Ncc)
                xc *= 2
                xc -= 1
                #x = np.random.randint(4, size=N)
                #x *= 2
                #x -= 1
                #x = np.where(x==3, -1j, x)
                #x = np.where(x==5, 1j, x)

                xc = xc.astype(A.dtype)

                x1 = cummP*xc
                x2 = cummR*x1

                # deflating Vx from x2
                x2_def = x2 - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x2))

                if use_Q:
                    if params['problem_name']=='schwinger':
                        x2_def = gamma3_application(x2_def)
                    elif params['problem_name']=='LQCD':
                        x2_def = gamma5_application(x2_def,nr_levels-1,params['dof'])

                #y,num_iters = solver_sparse(Ac,x2,level_solver_tol,ml_solvers[nr_levels-1],"mg")
                y = np.dot(np_Acc_inv,x2_def)
                y = np.asarray(y).reshape(-1)
                num_iters = 1

                output_params['results'][nr_levels-1]['function_iters'] += num_iters

                ests[i] = np.vdot(xc,y)

                # average of estimates
                ests_avg = np.sum(ests[0:(i+1)])/(i+1)
                # and standard deviation
                ests_dev = sqrt(np.sum(np.square(np.abs(ests[0:(i+1)]-ests_avg)))/(i+1))
                error_est = ests_dev/sqrt(i+1)

                #print(str(i)+" .. "+str(ests_avg)+" .. "+str(error_est)+" .. "+str(level_trace_tol))

                # break condition
                if i>=5 and error_est<level_trace_tol:
                    break

            output_params['results'][nr_levels-1]['nr_ests'] += i
            # set trace and standard deviation
            output_params['results'][nr_levels-1]['ests_avg'] = ests_avg+tr1c
            output_params['results'][nr_levels-1]['ests_dev'] = ests_dev

    end = time.time()
    print("\nTime to compute trace with MLMC (excluding rough trace and excluding setup time for the multigrid hierarchy) : "+str(end-start))

    if not (function_name=="exponential"):
        for i in range(nr_levels-1):
            #output_params['results'][i]['level_complexity'] += output_params['results'][i]['solver_iters']*ml_solvers[i].cycle_complexity()
            #print( "flops/cycle = "+str(flopsV(len(slvr.levels), slvr.levels, 0)) )
            if not params['problem_name']=='schwinger':
                slvr = ml_solvers[i]
                output_params['results'][i]['level_complexity'] = output_params['results'][i]['function_iters']*flopsV(len(slvr.levels), slvr.levels, 0)
            else:
                #output_params['results'][i]['level_complexity'] = output_params['results'][i]['function_iters']*flopsV_manual(len(mg.ml.levels), mg.ml.levels, i)
                output_params['results'][i]['level_complexity'] = output_params['results'][i]['function_iters']*flopsV_manual(i, mg.ml.levels, i)
                output_params['results'][i]['level_complexity'] += mg.ml.levels[len(mg.ml.levels)-1].A.nnz * mg.coarsest_lev_iters[i]

    if params['coarsest_level_directly']==True:
        output_params['results'][nr_levels-1]['level_complexity'] = pow(mg.ml.levels[nr_levels-1].A.shape[0],3) + \
                                                                    output_params['results'][nr_levels-1]['function_iters']*pow(mg.ml.levels[nr_levels-1].A.shape[0],2)
    else:
        output_params['results'][nr_levels-1]['level_complexity'] = output_params['results'][nr_levels-1]['function_iters']*(mg.ml.levels[nr_levels-1].A.shape[0]*mg.ml.levels[nr_levels-1].A.shape[0])

    #print( output_params['results'][nr_levels-1]['solver_iters'] * (ml.levels[nr_levels-1].A.shape[0]*ml.levels[nr_levels-1].A.shape[0]) )

    for i in range(nr_levels):
        output_params['total_complexity'] += output_params['results'][i]['level_complexity']

    # total trace
    for i in range(nr_levels):
        output_params['trace'] += output_params['results'][i]['ests_avg']

    print("")

    return output_params






# -----

    """
    #print(npnorm(A.todense()-ml.levels[0].A.todense(),'fro'))

    A1xxx = ml.levels[0].A.copy().todense()
    A2xxx = ml.levels[1].A.copy().todense()
    Q1 = A1xxx.copy()
    Q2 = A2xxx.copy()
    for i in range(Q1.shape[1]) : Q1[:,i] = gamma3_application(A1xxx[:,i])
    for i in range(Q2.shape[1]) : Q2[:,i] = gamma3_application(A2xxx[:,i])

    Bx1 = Q1.copy()
    Bx2 = Q2.copy()
    #Bx3 = ml.levels[2].A.copy().todense()

    Cx1 = ml.levels[0].P.copy()
    Dx1 = ml.levels[0].R.copy()
    #Cx2 = ml.levels[1].P.copy()
    #Dx2 = ml.levels[1].R.copy()

    Bx1inv = np.linalg.inv(Bx1)
    Bx2inv = np.linalg.inv(Bx2)
    #Bx3inv = np.linalg.inv(Bx3)

    Bx2invproj = Cx1*Bx2inv*Dx1
    #Bx3invproj = Cx2*Bx3inv*Dx2

    diffxinv1 = Bx1inv - Bx2invproj
    diffxinv1xx = diffxinv1 + diffxinv1.transpose()

    #diffxinv2 = Bx2inv - Bx3invproj
    #diffxinv2xx = diffxinv2 + diffxinv2.transpose()

    offdiag_fro_norm1 = npnorm( diffxinv1xx-np.diag(np.diag(diffxinv1xx)), ord='fro' )
    #offdiag_fro_norm2 = npnorm( diffxinv2xx-np.diag(np.diag(diffxinv2xx)), ord='fro' )

    print("\nTheoretical estimation of variance, diff at level 1 : "+str(0.5*offdiag_fro_norm1*offdiag_fro_norm1))
    #print("\nTheoretical estimation of variance, diff at level 2 : "+str(0.5*offdiag_fro_norm2*offdiag_fro_norm2))

    # -------------

    pure_x = Bx1inv+Bx1inv.transpose()
    offdiag_fro_norm = npnorm( pure_x-np.diag(np.diag(pure_x)), ord='fro' )
    print("\nTheoretical estimation of variance, pure : "+str(0.5*offdiag_fro_norm*offdiag_fro_norm))

    exit(0)
    """
