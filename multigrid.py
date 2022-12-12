import numpy as np
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import fgmres

from pyamg.aggregation.adaptive import adaptive_sa_solver
from scipy.sparse.linalg import eigsh, eigs
from math import pow, sqrt
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm
from numpy.linalg import norm as npnorm
from scipy.sparse import diags

from utils import CustomTimer

import matplotlib.pylab as plt




# ----------------------------------------------------------------------------------------------

# classes for multilevel information

class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

    # this operator permutes columns
    Pperm = 0
    perm_shift = 0
    Bblock_perm = 0

    g3 = 0

class SimpleML:
    def __init__(self):
        self.levels = []

    def __str__(self):
        for idx,level in enumerate(self.levels[:-1]):
            print("Level: "+str(idx))
            print("\tsize(R) = "+str(level.R.shape))
            print("\tsize(P) = "+str(level.P.shape))
            print("\tsize(A) = "+str(level.A.shape))




# ----------------------------------------------------------------------------------------------

# solver class
class MG:

    def __init__(self,A,smooth_iters=2):
        # This parameter changes in MLMC, and it indicates the level from which
        # we want to perform a solve
        self.level_nr = 0
        # This is the multigrid hierarchy, of type SimpleML
        self.ml = []
        self.A = A
        # Solution from the last call to the solver
        self.x = []
        self.num_iters = 0

        self.total_levels = 0
        self.coarsest_iters = 0
        self.coarsest_iters_tot = 0
        self.coarsest_iters_avg = 0
        self.nr_calls = 0

        self.smooth_iters = smooth_iters
        
        self.coarsest_lev_iters = [0,0,0,0,0,0,0,0,0,0]

        # This parameter allows us to run the method self.diff_op(...)
        # without having to use global variables
        self.level_for_diff_op = 0
        
        self.solve_tol = 1.0e-1
        
        self.coarsest_inv = []
        
        self.timer = CustomTimer()
        
        self.skip_level = False


    # <dof> :   per level (except the last one, of course), this is a list of
    #           the number of degrees of freedom for the next level
    # <aggrs> : per level, this is the block size. So, if at a certain level the
    #           value is 4, then the aggregates are of size 4^d, where d is the
    #           dimensionality of the physical problem

    # TODO : clean the code in this method !!

    def setup(self,dof=[2,8,8],aggrs=[2*2,2*2],max_levels=3,dim=2,acc_eigvs='low',sys_type='schwinger',params=None):

        # assuming a (roughly) minimum coarsest-level size for the matrix
        min_coarsest_size = 1

        # TODO : check what is the actual maximum number of levels possible. For
        #        now, just assume max_levels is possible

        Al = self.A.copy()

        As = list()
        Ps = list()
        Rs = list()

        # at level 0
        ml = SimpleML()
        ml.levels.append(LevelML())
        ml.levels[0].A = Al.copy()

        for i in range(max_levels-1):

            # these are actually the number of test vectors .. the use here is mixed
            # and should be changed
            if i==0:
                dofi = dof[i]
            else:
                dofi = int(dof[i]/2)
            dofip1 = int(dof[i+1]/2)

            # build gamma3 at each level
            diag_g3 = np.ones(Al.shape[0],dtype=Al.dtype)
            for ix in range(int(Al.shape[0]/2),Al.shape[0]):
                diag_g3[ix] = -diag_g3[ix]
            ml.levels[i].g3 = diags([diag_g3], [0])

            #for ix in range(4,12):
            #    print(ix)
            #plt.spy(ml.levels[i].g3)
            #plt.show()
            #exit(0)

            # construct permutation matrix at level 0
            if params['use_permuted'] and i==0:
                ndof = 2
                # extent of t dimension
                nt = params['latt_dims'][0]
                # displacement in the x dimension
                x_disp = params['x_displacement']
                # displacement in the rows of the matrix
                mat_disp = nt*ndof*x_disp
                ml.levels[0].perm_shift = mat_disp
                diagonals = [np.ones(Al.shape[0]-mat_disp), np.ones(mat_disp)]
                # in mg_solver.ml.levels[0].Pperm we save the matrix that permutes rows upwards
                ml.levels[0].Pperm = diags(diagonals, [-mat_disp, Al.shape[0]-mat_disp]).transpose()
                
                ml.levels[0].Bblock_perm = identity(Al.shape[0],dtype=Al.dtype)

            # use an eigensolver to compute the test vectors

            if params['test_vectors_type']=="LSVs" or params['test_vectors_type']=="RSVs":
                # use Q ... change before
                mat_size = int(Al.shape[0]/2)
                Al[mat_size:] = -Al[mat_size:]

            if acc_eigvs == 'low':
                tolx = tol=1.0e-3
                ncvx = dofip1+2
            elif acc_eigvs == 'high':
                tolx = tol=1.0e-9
                ncvx = None
            else:
                raise Exception("<accuracy_mg_eigvs> does not have a possible value.")

            if params['test_vectors_type']=="EVs":
                eigvals,eig_vecs = eigs( Al, k=dofip1, which='LM', tol=tolx, maxiter=1000000, sigma=0.0, ncv=ncvx )
            elif params['test_vectors_type']=="LSVs" or params['test_vectors_type']=="RSVs":
                Sy,eig_vecs = eigsh( Al,k=dofip1,which='LM',tol=tolx,sigma=0.0 )
            else:
                raise Exception("unknown type of test vectors")

            if params['test_vectors_type']=="LSVs" or params['test_vectors_type']=="RSVs":
                # use Q ... change before
                mat_size = int(Al.shape[0]/2)
                Al[mat_size:] = -Al[mat_size:]

            if params['test_vectors_type']=="LSVs":
                # transform into left singular vectors
                mat_size = int(eig_vecs.shape[0]/2)
                eig_vecs[mat_size:] = -eig_vecs[mat_size:]

            # now, construct P from the test vectors

            if i==0 : aggr_size = aggrs[i]*dofi
            else : aggr_size = aggrs[i]*dofi*2

            aggr_size_half = int(aggr_size/2)
            nr_aggrs = int(Al.shape[0]/aggr_size)
    
            P_size_n = Al.shape[0]
            P_size_m = nr_aggrs*dofip1*2
            Px = np.zeros((P_size_n,P_size_m), dtype=Al.dtype)

            # this is a for loop over aggregates
            for j in range(nr_aggrs):
                # this is a for loop over eigenvectors
                for k in range(dofip1):
                    # this is a for loop over half of the entries, spin 0
                    for w in range(int(aggr_size_half/(dofi/2))):
                        for z in range(int(dofi/2)):
                            # even entries
                            aggr_eigvectr_ptr = j*aggr_size+w*dofi+z
                            #ii_ptr = j*aggr_size+w
                            #ii_ptr = j*aggr_size+2*w
                            ii_ptr = j*aggr_size+w*dofi+z
                            jj_ptr = j*dofip1*2+k
                            Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

                # this is a for loop over eigenvectors
                for k in range(dofip1):
                    # this is a for loop over half of the entries, spin 1
                    for w in range(int(aggr_size_half/(dofi/2))):
                        for z in range(int(dofi/2)):
                            # odd entries
                            aggr_eigvectr_ptr = j*aggr_size+w*dofi+int(dofi/2)+z
                            #ii_ptr = j*aggr_size+aggr_size_half+w
                            ii_ptr = j*aggr_size+w*dofi+int(dofi/2)+z
                            jj_ptr = j*dofip1*2+dofip1+k
                            Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

            # ------------------------------------------------------------------------------------
            # perform a per-aggregate orthonormalization - apply plain CGS
            # spin 0
            for j in range(nr_aggrs):
                for k in range(dofip1):
                    ii_off_1 = j*aggr_size
                    #ii_off_2 = ii_off_1+aggr_size_half
                    ii_off_2 = ii_off_1+aggr_size
                    jj_off = j*dofip1*2
                    # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                    rs = []
                    for w in range(k):
                        rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                    for w in range(k):
                        Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                    Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
            # spin 1
            for j in range(nr_aggrs):
                for k in range(dofip1):
                    #ii_off_1 = j*aggr_size+aggr_size_half
                    #ii_off_2 = ii_off_1+aggr_size_half
                    ii_off_1 = j*aggr_size
                    ii_off_2 = ii_off_1+aggr_size
                    jj_off = j*dofip1*2+dofip1
                    # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                    rs = []
                    for w in range(k):
                        rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                    for w in range(k):
                        Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                    Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
            # ------------------------------------------------------------------------------------

            Pl = csr_matrix(Px, dtype=Px.dtype)

            ml.levels[i].P = Pl.copy()

            # set Rl = Pl^H
            Rl = Pl.copy()
            Rl = Rl.conjugate()
            Rl = Rl.transpose()
            
            #if params['use_permuted'] and i==0:
            #    Rl = Rl*ml.levels[0].Pperm

            ml.levels[i].R = Rl.copy()
    
            Ax = Rl*Al*Pl
            Al = Ax.copy()

            ml.levels.append(LevelML())
            ml.levels[i+1].A = Al.copy()

            if params['check_quality_MG']:

                #Pl2 = csr_matrix(Px, dtype=Px.dtype)
                #write_png(Pl2,"P2_"+str(i)+".png")

                # check Gamma3-compability here
                P1 = np.copy(Px)
                mat_size1_half = int(P1.shape[0]/2)
                P1[mat_size1_half:,:] = -P1[mat_size1_half:,:]
                P2 = np.copy(Px)
                mat_size2_half = int(P1.shape[1]/2)
                P2[:,mat_size2_half:] = -P2[:,mat_size2_half:]
                diffP = P1-P2
                print("\tmeasuring g3-compatibility at level "+str(i)+" : "+str( npnorm(diffP,ord='fro') ))

                axx = Rl*Pl
                bxx = identity(Pl.shape[1],dtype=Pl.dtype)
                cxx = axx-bxx
                print("\torthonormality of P at level "+str(i)+" = "+str( norm(axx-bxx,ord='fro')) )

                print("\tconstructing A at level "+str(i+1)+" ...")

                Ax = Ax.getH()
                print("\thermiticity of A at level "+str(i+1)+" = "+str( norm(Ax-Al,ord='fro')) )

                mat_size_half = int(Al.shape[0]/2)
                g3Al = Al.copy()
                g3Al[mat_size_half:,:] = -g3Al[mat_size_half:,:]
                g3Ax = g3Al.copy()
                g3Ax = g3Ax.getH()
                print("\thermiticity of g3*A at level "+str(i+1)+" = "+str( norm(g3Ax-g3Al,ord='fro')) )

                print("\t... done")

                if Al.shape[0] <= min_coarsest_size: break

            # some permuted matrix checks

            if params["use_permuted"]:
                mat_disp =  int(( ml.levels[i].perm_shift / ( dof[i]*aggrs[i] ) ) * dof[i+1])

                ml.levels[i+1].perm_shift = mat_disp
                diagonals = [np.ones(Pl.shape[1]-mat_disp), np.ones(mat_disp)]
                # in mg_solver.ml.levels[i].Pperm we save the matrix that permutes rows upwards
                ml.levels[i+1].Pperm = diags(diagonals, [-mat_disp, Pl.shape[1]-mat_disp]).transpose()

                Bl = ml.levels[i].Pperm.transpose().conjugate() * (Pl*ml.levels[i+1].Pperm)
                #Bl = ml.levels[i].Pperm.transpose().conjugate() * (Pl)
                Bl = (Rl*ml.levels[i].Bblock_perm) * Bl
                ml.levels[i+1].Bblock_perm = Bl

        # creating Q -- Schwinger specific
        #for i in range(len(ml.levels)):
        #    half_size = int(ml.levels[i].A.shape[0]/2)
        #    ml.levels[i].Q = ml.levels[i].A.copy()
        #    ml.levels[i].Q[mat_size_half:,:] = -ml.levels[i].Q[mat_size_half:,:]

        self.ml = ml

        # pre-compute the inverse of the coarsest-level matrix
        Acc = self.ml.levels[len(self.ml.levels)-1].A
        np_Acc = Acc.todense()
        self.coarsest_inv = np.linalg.inv(np_Acc)
        

    def solve(self,A,b,tol):

        num_iters = 0
        def callback(xk):
            nonlocal num_iters
            num_iters += 1

        if A.shape[0]<1000:
            maxiter = A.shape[0]
        else:
            maxiter = 1000

        self.A = self.ml.levels[self.level_nr].A
        lop1 = LinearOperator(A.shape, matvec=self.matvec)
        lop2 = LinearOperator(A.shape, matvec=self.one_mg_step)
        self.x,exitCode = fgmres(lop1,b,tol=tol,M=lop2,callback=callback,maxiter=maxiter)
        
        #print("solver iters = "+str(num_iters))

        self.num_iters = num_iters


    def one_mg_step(self,b):

        level_id = self.total_levels-self.level_nr

        rs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) \
               for i in range(self.level_nr,self.total_levels) ]
        bs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) \
               for i in range(self.level_nr,self.total_levels) ]
        xs = [ np.zeros(self.ml.levels[i].A.shape[0],dtype=self.ml.levels[i].A.dtype) \
               for i in range(self.level_nr,self.total_levels) ]

        self.timer.start("axpy")
        bs[0][:] = b[:]
        self.timer.end("axpy")
        
        # go down in the V-cycle
        for i in range(level_id-1):
            # 1. build the residual
            self.timer.start("mvm")
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]
            self.timer.end("mvm")
            # 2. smooth
            self.A = self.ml.levels[i+self.level_nr].A
            lop = LinearOperator(self.A.shape, matvec=self.matvec)
            e, exitCode = lgmres( lop,rs[i],tol=1.0e-20, \
                                  maxiter=self.smooth_iters )
            self.A = self.ml.levels[self.level_nr].A
            # 3. update solution
            self.timer.start("axpy")
            xs[i] += e
            self.timer.end("axpy")
            # 4. update residual
            self.timer.start("mvm")
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]
            self.timer.end("mvm")
            # 5. restrict residual
            self.timer.start("R")
            bs[i+1] = self.ml.levels[i+self.level_nr].R*rs[i]
            self.timer.end("R")
    
        # coarsest level solve

        i += 1

        self.timer.start("mvm")
        y = np.dot(self.coarsest_inv,bs[i])
        xs[i] = np.asarray(y).reshape(-1)
        self.timer.end("mvm")
        num_iters = 1

        self.coarsest_lev_iters[self.level_nr] += num_iters
        self.coarsest_iters = num_iters
        self.nr_calls += 1
        self.coarsest_iters_tot += self.coarsest_iters
        self.coarsest_iters_avg = self.coarsest_iters_tot/self.nr_calls
    
        # go up in the V-cycle
        for i in range(level_id-2,-1,-1):
            # 1. interpolate and update
            self.timer.start("P")
            xs[i] += self.ml.levels[i+self.level_nr].P*xs[i+1]
            self.timer.end("P")
            # 2. build the residual
            self.timer.start("mvm")
            rs[i] = bs[i]-self.ml.levels[i+self.level_nr].A*xs[i]
            self.timer.end("mvm")
            # 3. smooth
            self.A = self.ml.levels[i+self.level_nr].A
            lop = LinearOperator(self.A.shape, matvec=self.matvec)
            e, exitCode = lgmres( lop,rs[i],tol=1.0e-20, \
                                  maxiter=self.smooth_iters )
            self.A = self.ml.levels[self.level_nr].A

            # 4. update solution
            self.timer.start("axpy")
            xs[i] += e
            self.timer.end("axpy")

        return xs[0]


    def __str__(self):
        str_out = ""
        str_out += "\nMultilevel information:\n"
        for idx,level in enumerate(self.ml.levels):
            str_out += "Level: "+str(idx)+"\n"
            if idx<(len(self.ml.levels)-1): str_out += "\tsize(R) = "+str(level.R.shape)+"\n"
            if idx<(len(self.ml.levels)-1): str_out += "\tsize(P) = "+str(level.P.shape)+"\n"
            str_out += "\tsize(A) = "+str(level.A.shape)+"\n"
        return str_out


    def diff_op_Q(self,v):

        # apply g5
        v_size = int(v.shape[0]/2)
        vx = v[:]
        vx[v_size:] = -vx[v_size:]

        return self.diff_op(vx)


    def diff_op(self,v):

        # this function applies:
        # ( Af^{-1} - P*Ac^{-1}*R ) * g5 * v

        """
        example of skipping a level:
        
        A0inv - P0*A1inv*R0
        A1inv - P1*A2inv*R1
        A2inv - P2*A3inv*R2
        A3inv

        A0inv - P0*P1*A2inv*R1*R0
        A2inv - P2*A3inv*R2
        A3inv
        """

        level_nr = self.level_for_diff_op

        v_size = int(v.shape[0]/2)
        vx = v[:]

        #vx[v_size:] = -vx[v_size:]

        if self.skip_level and level_nr==0:
            Af = self.ml.levels[level_nr].A
            Ac = self.ml.levels[level_nr+1+1].A
            P0 = self.ml.levels[level_nr].P
            R0 = self.ml.levels[level_nr].R
            P1 = self.ml.levels[level_nr+1].P
            R1 = self.ml.levels[level_nr+1].R
        else:
            Af = self.ml.levels[level_nr].A
            Ac = self.ml.levels[level_nr+1].A
            P = self.ml.levels[level_nr].P
            R = self.ml.levels[level_nr].R

        vf = vx
        self.timer.start("R")
        if self.skip_level and level_nr==0:
            vc = R1*(R0*vx)
        else:
            vc = R*vx
        self.timer.end("R")

        self.level_nr = level_nr
        self.solve(Af,vf,self.solve_tol)
        t1 = self.x

        if self.skip_level and level_nr==0:
            if (level_nr+2)==(len(self.ml.levels)-1):
                self.timer.start("mvm")
                t2 = np.dot(self.coarsest_inv,vc)
                self.timer.end("mvm")
                t2 = np.asarray(t2).reshape(-1)
            else:
                self.level_nr = level_nr+1+1
                self.solve(Ac,vc,self.solve_tol)
                t2 = self.x
        else:
            if (level_nr+1)==(len(self.ml.levels)-1):
                self.timer.start("mvm")
                t2 = np.dot(self.coarsest_inv,vc)
                self.timer.end("mvm")
                t2 = np.asarray(t2).reshape(-1)
            else:
                self.level_nr = level_nr+1
                self.solve(Ac,vc,self.solve_tol)
                t2 = self.x

        self.timer.start("P")
        if self.skip_level and level_nr==0:
            vout = t1 - P0*(P1*t2)
        else:
            vout = t1 - P*t2
        self.timer.end("P")

        return vout


    def matvec(self,x):
        
        self.timer.start("mvm")
        y = self.A*x
        self.timer.end("mvm")
        return y
