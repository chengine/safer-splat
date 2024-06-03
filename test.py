#%%
import numpy as np
import cvxpy as cvx

# Section 8.4.1, Boyd & Vandenberghe "Convex Optimization"
# Original version by Lieven Vandenberghe
# Updated for CVX by Almir Mutapcic - Jan 2006
# (a figure is generated)
#
# We find a smallest ellipsoid containing m ellipsoids
# [ x'*A_i*x + 2*b_i'*x + c < 0 ], for i = 1,...,m
#
# Problem data:
# As = [A1, A2, ..., Am]:  cell array of m pos. def. matrices
# bs = [b1, b2, ..., bm]:  cell array of m 2-vectors
# cs = [c1, c2, ..., cm]:  cell array of m scalars

# ellipse data
n= 3
m = 10
As = np.random.randn(m, n, n)
As = np.matmul(As, As.transpose(0,2,1))
bs = np.random.randn(m, n)
cs = np.random.randn(m)

Asqr = cvx.Variable((n,n), symmetric=True) 
btilde = cvx.Variable(n)
t = cvx.Variable(m)
objective = cvx.Maximize( cvx.log_det(Asqr) )
constraints = [t >= 0]
for i in range(m):
    mat1 = np.hstack([Asqr-t[i]*As[i], cvx.reshape((btilde - t[i]*bs[i]), (n, 1)), np.zeros((n, n))])
    mat2 = np.hstack([(btilde - t[i]*bs[i]).T, -1 - t[i]*cs[i], btilde])
    mat3 = np.hstack([np.zeros((n, n)), btilde.T, -Asqr])

    constraints += [np.vstack([mat1, mat2, mat3])>= 0]


# convert to ellipsoid parametrization E = [ x | || Ax + b || <= 1 ]
