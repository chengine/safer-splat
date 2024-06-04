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
        self.beta = alpha
        self.rel_deg = dynamics.rel_deg

        # Create an OSQP object
        self.prob = osqp.OSQP()

        self.times_solved = 0

    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        h, grad_h, hes_h = self.gsplat.query_distance(x)       # can pass in an optional argument for a radius


        # h is 1 x 1, grad_h is 1 x 3, hes_h is 3 x 3

        # we need h to be 1x1, grad_h to be 1x6, hes_h to be 6x6
        # add zeros to the right of grad_h
        grad_h = torch.cat((grad_h, torch.zeros(1, 3).to(grad_h.device)), dim=-1)

        # add zeros to the right of hes_h
        hes_h = torch.cat((hes_h, torch.zeros(3, 3).to(hes_h.device)), dim=-1)



        f, g, df = self.dynamics.system(x)

        # f is 6x1, g is 6x1, df is 6x6

        # Extended barrier function
        lfh = torch.matmul(grad_h, f[None]).squeeze()



        # lgh = torch.matmul(grad_h, g[None]).squeeze() # this is equal to 0

        # Compute the Lie derivatives of the Lie derivatives
        # lflfh is (d2h/dx2 * f(x) + dh/dx * df/dx) * f(x)
        # lglfh is (d2h/dx2 * f(x) + dh/dx * df/dx) * g(x)

        # need to check below

        lflfh = torch.matmul(torch.matmul(hes_h, f.unsqueeze(-1)), f.unsqueeze(0)).squeeze() + torch.matmul(grad_h, torch.matmul(df, f.unsqueeze(-1))).squeeze()
        lglfh = torch.matmul(torch.matmul(hes_h, f.unsqueeze(-1)), g.unsqueeze(0)).squeeze() + torch.matmul(grad_h, torch.matmul(df, g.unsqueeze(-1))).squeeze()


        # our full constraint is
        # lflfh + lglfh * u + alpha(lfh) + beta(lfh + alpha(h)) <= 0

        l = -lflfh - self.alpha(lfh) - self.beta(lfh + self.alpha(h))
        A = lglfh[None]  # 1 x 6

        P = torch.eye(3).to(grad_h.device)
        qt = torch.tensor(u_des).to(grad_h.device)

        # grad_h_f = torch.sum(grad_h * f[None], dim=-1).squeeze()
        # grad_h_g = torch.matmul(grad_h, g[None]).squeeze()

        # alpha_h = self.alpha(h-1)

        
        # b = -alpha_h - grad_h_f - A @ u_des

        # if A.dim() == 1:
        #     A = A[None]
        # # We want to solve for a minimal set of constraints in the Polytope
        # # First, normalize
        # Anorm = torch.norm(A, dim=1)
        # A = A / Anorm[:, None]
        # b = b / Anorm

        # A = A.cpu().numpy()
        # b = b.cpu().numpy()
        # if minimal:
        #     # Need to pass in an interior point to the polytope
        #     pt = find_interior(A, b)
        #     A, b = h_rep_minimal(A, b, pt)

        A = A.cpu().numpy()
        l = l.cpu().numpy()
        P = P.cpu().numpy()
        qt = qt.cpu().numpy()

        return A, l, P, qt

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
            self.prob.setup(P=P, A=A, l=b)
        else:
            self.prob.update(Ax=A.data, l=b)
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

    