import torch
from ellipsoids.math_utils import h_rep_minimal, find_interior
import osqp
import numpy as np
from scipy import sparse
import cvxpy as cvx

class CBF():
    def __init__(self, gsplat, dynamics, alpha):
        # gsplat: GSplat object
        # dynamics: function that returns f, g given x
        # alpha: class K extended function

        self.gsplat = gsplat
        self.dynamics = dynamics
        self.alpha = alpha

        # Create an OSQP object
        self.prob = osqp.OSQP()

        self.times_solved = 0

    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        h, grad_h = self.gsplat.query_distance(x)       # can pass in an optional argument for a radius
        f, g = self.dynamics.system(x)

        grad_h_f = torch.sum(grad_h * f[None], dim=-1).squeeze()
        grad_h_g = torch.matmul(grad_h, g[None]).squeeze()

        alpha_h = self.alpha(h)

        A = grad_h_g
        b = -alpha_h - grad_h_f - A @ u_des

        if A.dim() == 1:
            A = A[None]
        # We want to solve for a minimal set of constraints in the Polytope
        # First, normalize
        Anorm = torch.norm(A, dim=1)
        A = A / Anorm[:, None]
        b = b / Anorm

        A = A.cpu().numpy()
        b = b.cpu().numpy()
        if minimal:
            # Need to pass in an interior point to the polytope
            pt = find_interior(A, b)
            A, b = h_rep_minimal(A, b, pt)

        return A, b

    def solve_QP(self, x, u_des):
        A, b = self.get_QP_matrices(x, u_des, minimal=False)
        
        p = self.optimize_QP(A, b)       # Need to fill this out

        # return the optimal control
        u = torch.tensor(p).to(device=u_des.device, dtype=torch.float32) + u_des

        return u

    def optimize_QP(self, A, b):
        udim = A.shape[1]

        # Setup workspace
        P = sparse.eye(udim)
        A = sparse.csc_matrix(A)

        if self.times_solved == 0:
            self.prob.setup(P=P, A=A, u=b)
        else:
            self.prob.update(Ax=A.data, u=b)
        self.times_solved += 1

        # Solve
        res = self.prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        output = res.x

        # p = cvx.Variable(udim)
        # objective = cvx.Minimize(cvx.sum_squares(p))
        # constraints = [A @ p <= b]
        # prob = cvx.Problem(objective, constraints)
        # prob.solve()
        # output = p.value
        
        return output

    