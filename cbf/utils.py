import torch
from ellipsoids.math_utils import h_rep_minimal, find_interior

class CBF():
    def __init__(self, gsplat, dynamics, alpha):
        # gsplat: GSplat object
        # dynamics: function that returns f, g given x
        # alpha: class K extended function

        self.gsplat = gsplat
        self.dynamics = dynamics
        self.alpha = alpha

    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        h, grad_h = self.gsplat.query_distance(x)       # can pass in an optional argument for a radius
        f, g = self.dynamics(x)

        grad_h_f = torch.sum(grad_h * f[None], dim=-1).squeeze()
        grad_h_g = torch.matmul(grad_h, g[None]).squeeze()

        alpha_h = self.alpha(h)

        A = grad_h_g
        b = -alpha_h - grad_h_f - A @ u_des

        # We want to solve for a minimal set of constraints in the Polytope
        if minimal:
            A = A.cpu().numpy()
            b = b.cpu().numpy()

            # Need to pass in an interior point to the polytope
            pt = find_interior(A, b)
            A, b = h_rep_minimal(A, b, pt)

        return A, b

    def solve_QP(self, x, u_des):
        A, b = self.get_QP_matrices(x, u_des, minimal=True)
        
        p = optimize(...)       # Need to fill this out

        # return the optimal control
        u = p + u_des

        return u


    