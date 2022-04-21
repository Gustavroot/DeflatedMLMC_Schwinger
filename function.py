import scipy as sp
import pyamg
import numpy as np
import warnings

import pymatlab

from scipy.sparse import csc_matrix


# TODO : identify what would be the equivalent to num_iters here
mscript = """

%function [w, err, hump] = expv( t, A, v, tol, m )

A = sparse(Ax);

t = -1.0;

tolx = 1.0e-7;
m = 30;

[n,n] = size(A);

%if nargin == 3,
%  tol = 1.0e-7;
%  m = min(n,30);
%end;
%if nargin == 4,
%  m = min(n,30);
%end;

anorm = norm(A,'inf'); 
mxrej = 10;  btol  = 1.0e-7; 
gamma = 0.9; delta = 1.2; 
mb    = m; t_out   = abs(t);
nstep = 0; t_new   = 0;
t_now = 0; s_error = 0;
rndoff= anorm*eps;

k1 = 2; xm = 1/m; normv = norm(v); beta = normv;
fact = (((m+1)/exp(1))^(m+1))*sqrt(2*pi*(m+1));
t_new = (1/anorm)*((fact*tolx)/(4*beta*anorm))^xm;
s = 10^(floor(log10(t_new))-1); t_new = ceil(t_new/s)*s; 
sgn = sign(t); nstep = 0;

w = v;
hump = normv;
while t_now < t_out
  nstep = nstep + 1;
  t_step = min( t_out-t_now,t_new );
  V = zeros(n,m+1); 
  H = zeros(m+2,m+2);

  V(:,1) = (1/beta)*w;
  for j = 1:m
     p = A*V(:,j);
     for i = 1:j
        H(i,j) = V(:,i)'*p;
        p = p-H(i,j)*V(:,i);
     end;
     s = norm(p); 
     if s < btol,
        k1 = 0;
        mb = j;
        t_step = t_out-t_now;
        break;
     end;
     H(j+1,j) = s;
     V(:,j+1) = (1/s)*p;
  end; 
  if k1 ~= 0, 
     H(m+2,m+1) = 1;
     avnorm = norm(A*V(:,m+1)); 
  end;
  ireject = 0;
  while ireject <= mxrej,
     mx = mb + k1;
     F = expm(sgn*t_step*H(1:mx,1:mx));
     if k1 == 0,
	err_loc = btol; 
        break;
     else
        phi1 = abs( beta*F(m+1,1) );
        phi2 = abs( beta*F(m+2,1) * avnorm );
        if phi1 > 10*phi2,
           err_loc = phi2;
           xm = 1/m;
        elseif phi1 > phi2,
           err_loc = (phi1*phi2)/(phi1-phi2);
           xm = 1/m;
        else
           err_loc = phi1;
           xm = 1/(m-1);
        end;
     end;
     if err_loc <= delta * t_step*tolx,
        break;
     else
        t_step = gamma * t_step * (t_step*tolx/err_loc)^xm;
        s = 10^(floor(log10(t_step))-1);
        t_step = ceil(t_step/s) * s;
        if ireject == mxrej,
           error('The requested tolerance is too high.');
        end;
        ireject = ireject + 1;
     end;
  end;
  mx = mb + max( 0,k1-1 );
  w = V(:,1:mx)*(beta*F(1:mx,1));
  beta = norm( w );
  hump = max(hump,beta);

  t_now = t_now + t_step;
  t_new = gamma * t_step * (t_step*tolx/err_loc)^xm;
  s = 10^(floor(log10(t_new))-1); 
  t_new = ceil(t_new/s) * s;

  err_loc = max(err_loc,rndoff);
  s_error = s_error + err_loc;

end;

err = s_error;
hump = hump / normv;

"""

#mscript = 'B = A; \
#for i=1:10 \
#    B = 2*B; \
#end \
#'

def function_sparse(A, b, tol, function, function_name, spec_name):
    num_iters = 0

    if not (function_name=="exponential"):

        warnings.simplefilter("ignore")

        #def callback(xk):
        #    class context:
        #        y = 0
        #    def inner():
        #        context.y += 1
        #        return context.y
        #    return inner

        #"""
        def callback(xk):
            nonlocal num_iters
            num_iters += 1
        #"""

        # call the function on your data
        if spec_name=="cg" or spec_name=="gmres":
            x,_ = function(A, b, callback=callback, tol=tol)
        elif spec_name=="mg":
            #x = function.solve(b, accel='cg', callback=callback)
            x = function.solve(b, accel='gmres', callback=callback, tol=tol, maxiter=b.shape[0])
            #x = solver.solve(b, callback=callback)

            #print(tol)

            #rel_res = np.linalg.norm(b - A*x)/np.linalg.norm(b)

            #print(rel_res)
            #print(np.linalg.norm(b))
            #print(np.linalg.norm(x))
            #print(num_iters)
            #exit(0)

        #num_iters = callback(0)()
        #print(num_iters)
        #num_iters = callback()

    else:

        function.putvalue('v',b)

        #session.putvalue('t',t)
        function.run(mscript)

        #session.putvalue('MSCRIPT',mscript)
        #session.run('eval(MSCRIPT)')

        x = function.getvalue('w')
        err = function.getvalue('err')
        hump = function.getvalue('hump')

    return (x,num_iters)


def loadFunction(spec_name, A=None, B=None, function_name='inverse'):

    if not (function_name=="exponential"):

        # FIXME
        max_nr_levels = 9

        if spec_name=='cg':
           return sp.sparse.linalg.cg

        elif spec_name=='gmres':
            return sp.sparse.linalg.gmres

        elif spec_name=='mg':
            [ml, work] = pyamg.aggregation.adaptive.adaptive_sa_solver(A, num_candidates=1, candidate_iters=1, improvement_iters=1, strength='symmetric', aggregate='standard', max_levels=2)
            return ml

        else:
            raise Exception("From <loadFunction(...)> : the chosen <spec_name> is not supported.")

    else:
        print("Creating MATLAB session ...")
        session = pymatlab.session_factory()
        print("... done")
        return session
