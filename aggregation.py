from pyamg.aggregation.adaptive import adaptive_sa_solver
from scipy.sparse.linalg import eigsh, eigs
from math import pow, sqrt
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import norm
from numpy.linalg import norm as npnorm
#import png





class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        for idx,level in enumerate(self.levels[:-1]):
            print("Level: "+str(idx))
            print("\tsize(R) = "+str(level.R.shape))
            print("\tsize(P) = "+str(level.P.shape))
            print("\tsize(A) = "+str(level.A.shape))


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


# in LQCD e.g. (4^4 lattice) :
#	manual_aggregation( A, [12,24], [2*2*2*2] )

# in LQCD e.g. (8^4 lattice) :
#	manual_aggregation( A, [12,24], [4*4*4*4] )



# <dof> :   per level (except the last one, of course), this is a list of
#           the number of degrees of freedom for the next level
# <aggrs> : per level, this is the block size. So, if at a certain level the
#           value is 4, then the aggregates are of size 4^d, where d is the
#           dimensionality of the physical problem
def manual_aggregation(A, dof=[2,2,2], aggrs=[2*2,2*2], max_levels=3, dim=2, acc_eigvs='low', sys_type='schwinger'):

    # assuming a (roughly) minimum coarsest-level size for the matrix
    min_coarsest_size = 1

    # TODO : check what is the actual maximum number of levels possible. For
    #        now, just assume max_levels is possible

    Al = A.copy()

    As = list()
    Ps = list()
    Rs = list()

    # at level 0
    ml = SimpleML()
    ml.levels.append(LevelML())
    ml.levels[0].A = Al.copy()

    print("")

    for i in range(max_levels-1):

        # use Q
        #mat_size = int(Al.shape[0]/2)
        #Al[mat_size:] = -Al[mat_size:]

        print("\tNonzeros = "+str(Al.count_nonzero()))
        print("\tsize(A) = "+str(Al.shape))

        print("\teigensolving at level "+str(i)+" ...")

        nt = 1

        if acc_eigvs == 'low':
            tolx = tol=1.0e-1
            ncvx = nt*dof[i+1]+2
        elif acc_eigvs == 'high':
            tolx = tol=1.0e-5
            ncvx = None
        else:
            raise Exception("<accuracy_mg_eigvs> does not have a possible value.")

        # FIXME : hardcoded value for eigensolving tolerance for now
        tolx = 1.0e-5

        #eigvals,eig_vecsx = eigsh(Al, k=nt*dof[i+1], which='SM', return_eigenvectors=True, tol=1e-5, maxiter=1000000)
        #eigvals,eig_vecsx = eigs(Al, k=nt*dof[i+1], which='SM', return_eigenvectors=True, tol=1e-2, maxiter=1000000)

        #eigvals,eig_vecsx = eigsh( Al, k=1, which='SR', tol=tolx, maxiter=1000000 )
        #print(eigvals)
        #exit(0)

        if i<3:
            eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=tolx, maxiter=1000000, sigma=0.0, ncv=ncvx )
            #eigvals,eig_vecsx = eigsh( Al, k=nt*dof[i+1], which='SM', tol=tolx, maxiter=1000000 )
        else:
            #eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=1.0e-5, maxiter=1000000, sigma=0.0 )
            eigvals,eig_vecsx = eigs( Al, k=nt*dof[i+1], which='LM', tol=tolx, maxiter=1000000, sigma=0.0 )
            #eigvals,eig_vecsx = eigsh( Al, k=nt*dof[i+1], which='SM', tol=1.0e-5, maxiter=1000000 )

        #print(eigvals)
        #exit(0)

        eig_vecs = np.zeros((Al.shape[0],dof[i+1]), dtype=Al.dtype)

        #coeffs = [ 1.0/float(k+1) for k in range(nt) ]
        coeffs = [ 1.0 for k in range(nt) ]

        for j in range(dof[i+1]):
            for k in range(nt):
                eig_vecs[:,j] += eig_vecsx[:,nt*j+k]
                #eig_vecs[:,j] += coeffs[k]*eig_vecsx[:,j+dof[i+1]*k]

        print("\t... done")

        # use Q
        #mat_size = int(Al.shape[0]/2)
        #Al[mat_size:] = -Al[mat_size:]

        print("\tconstructing P at level "+str(i)+" ...")

        #aggr_size = aggrs[i]*aggrs[i]*dof[i]

        if i==0 : aggr_size = aggrs[i]*dof[i]
        else : aggr_size = aggrs[i]*dof[i]*2

        aggr_size_half = int(aggr_size/2)
        nr_aggrs = int(Al.shape[0]/aggr_size)

        P_size_n = Al.shape[0]
        P_size_m = nr_aggrs*dof[i+1]*2
        Px = np.zeros((P_size_n,P_size_m), dtype=Al.dtype)

        # this is a for loop over aggregates
        for j in range(nr_aggrs):
            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):
                # this is a for loop over half of the entries, spin 0
                for w in range(int(aggr_size_half/(dof[i]/2))):
                    for z in range(int(dof[i]/2)):
                        # even entries
                        aggr_eigvectr_ptr = j*aggr_size+w*dof[i]+z
                        #ii_ptr = j*aggr_size+w
                        #ii_ptr = j*aggr_size+2*w
                        ii_ptr = j*aggr_size+w*dof[i]+z
                        jj_ptr = j*dof[i+1]*2+k
                        Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

            # this is a for loop over eigenvectors
            for k in range(dof[i+1]):
                # this is a for loop over half of the entries, spin 1
                for w in range(int(aggr_size_half/(dof[i]/2))):
                    for z in range(int(dof[i]/2)):
                        # odd entries
                        aggr_eigvectr_ptr = j*aggr_size+w*dof[i]+int(dof[i]/2)+z
                        #ii_ptr = j*aggr_size+aggr_size_half+w
                        ii_ptr = j*aggr_size+w*dof[i]+int(dof[i]/2)+z
                        jj_ptr = j*dof[i+1]*2+dof[i+1]+k
                        Px[ii_ptr,jj_ptr] = eig_vecs[aggr_eigvectr_ptr,k]

        print("\t... done")

        # ------------------------------------------------------------------------------------
        # perform a per-aggregate orthonormalization - apply plain CGS
        print("\torthonormalizing by aggregate in P at level "+str(i)+" ...")
        # spin 0
        for j in range(nr_aggrs):
            for k in range(dof[i+1]):
                ii_off_1 = j*aggr_size
                #ii_off_2 = ii_off_1+aggr_size_half
                ii_off_2 = ii_off_1+aggr_size
                jj_off = j*dof[i+1]*2
                # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                rs = []
                for w in range(k):
                    rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                for w in range(k):
                    Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
        # spin 1
        for j in range(nr_aggrs):
            for k in range(dof[i+1]):
                #ii_off_1 = j*aggr_size+aggr_size_half
                #ii_off_2 = ii_off_1+aggr_size_half
                ii_off_1 = j*aggr_size
                ii_off_2 = ii_off_1+aggr_size
                jj_off = j*dof[i+1]*2+dof[i+1]
                # vk = Px[ii_off_1:ii_off_2,jj_off+k]
                rs = []
                for w in range(k):
                    rs.append(np.vdot(Px[ii_off_1:ii_off_2,jj_off+w],Px[ii_off_1:ii_off_2,jj_off+k]))
                for w in range(k):
                    Px[ii_off_1:ii_off_2,jj_off+k] -= rs[w]*Px[ii_off_1:ii_off_2,jj_off+w]
                Px[ii_off_1:ii_off_2,jj_off+k] /= sqrt(np.vdot(Px[ii_off_1:ii_off_2,jj_off+k],Px[ii_off_1:ii_off_2,jj_off+k]))
        print("\t... done")
        # ------------------------------------------------------------------------------------

        Pl = csr_matrix(Px, dtype=Px.dtype)

        ml.levels[i].P = Pl.copy()

        print("\tconstructing R at level "+str(i)+" ...")

        # set Rl = Pl^H
        Rl = Pl.copy()
        Rl = Rl.conjugate()
        Rl = Rl.transpose()

        print("\t... done")

        ml.levels[i].R = Rl.copy()

        Ax = Rl*Al*Pl
        Al = Ax.copy()

        ml.levels.append(LevelML())
        ml.levels[i+1].A = Al.copy()

        """
        if sys_type=='schwinger':

            #Pl2 = csr_matrix(Px, dtype=Px.dtype)
            #write_png(Pl2,"P2_"+str(i)+".png")

            # check Gamma3-compability here !!
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

        print("")
        """

    print("\tNonzeros = "+str(Al.count_nonzero()))
    print("\tsize(A) = "+str(Al.shape))

    # creating Q -- Schwinger specific
    #for i in range(len(ml.levels)):
    #    half_size = int(ml.levels[i].A.shape[0]/2)
    #    ml.levels[i].Q = ml.levels[i].A.copy()
    #    ml.levels[i].Q[mat_size_half:,:] = -ml.levels[i].Q[mat_size_half:,:]

    return ml
