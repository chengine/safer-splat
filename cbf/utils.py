import torch
import osqp
import numpy as np
from scipy import sparse

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


        h = h.unsqueeze(-1)
        grad_h = torch.cat((grad_h, torch.zeros(h.shape[0], 3).to(grad_h.device)), dim=-1)
        # add zeros to the right of hes_h
        hes_h = torch.cat((hes_h, torch.zeros(h.shape[0], 3, 3).to(hes_h.device)), dim=2)
        hes_h = torch.cat((hes_h, torch.zeros(h.shape[0], 3, 6).to(hes_h.device)), dim=1)

        # # print x shape
        # print(f"x shape: {x.shape}")

        # # print shape of hessian grad and h
        # print(f"hes_h shape: {hes_h.shape}")
        # print(f"grad_h shape: {grad_h.shape}")
        # print(f"h shape: {h.shape}")


        f, g, df = self.dynamics.system(x)

        f = f.unsqueeze(-1)
        # f is 6x1, g is 6x1, df is 6x6
        #print shapes of f g df
        # print(f"f shape: {f.shape}")
        # print(f"g shape: {g.shape}")
        # print(f"df shape: {df.shape}")


        # Extended barrier function
        lfh = torch.matmul(grad_h, f).squeeze()



        # Compute the Lie derivatives of the Lie derivatives
        # lflfh is (d2h/dx2 * f(x) + dh/dx * df/dx) * f(x)
        # lglfh is (d2h/dx2 * f(x) + dh/dx * df/dx) * g(x)

        # need to check below

        lflfh = torch.matmul(f.T, torch.matmul(hes_h, f)).squeeze() + torch.matmul(grad_h, torch.matmul(df, f)).squeeze()
        lglfh = torch.matmul(g.T, torch.matmul(hes_h, f)).squeeze() + torch.matmul(grad_h, torch.matmul(df, g)).squeeze()

        # our full constraint is
        # lflfh + lglfh * u + alpha(lfh) + beta(lfh + alpha(h)) <= 0

        l = -lflfh - self.alpha(lfh) - self.beta(lfh + self.alpha(h.squeeze()))
        A = lglfh[None]  # 1 x 6

        P = torch.eye(3).to(grad_h.device)
        qt = torch.tensor(u_des).to(grad_h.device)

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

        A = A.cpu().numpy().squeeze()
        l = l.cpu().numpy()
        qt = qt.cpu().numpy()

        return A, l

    def solve_QP(self, x, u_des):
        A, l = self.get_QP_matrices(x, u_des, minimal=False)
    
        # lets use random subset of the constraints
        A = A[:3]
        l = l[:3]


        q = u_des.cpu().numpy()
        p = self.optimize_QP(A, l, q)       # Need to fill this out

        # return the optimal control
        u = torch.tensor(p).to(device=u_des.device, dtype=torch.float32) + u_des

        return u

    def optimize_QP(self, A, l, q):
        udim = A.shape[1]

        # Setup workspace
        P = sparse.eye(udim)
        A = sparse.csc_matrix(A)

        if self.times_solved == 0:
            self.prob.setup(P=P, A=A, l=l, q=q, verbose=False)
        else:
            self.prob.update(Ax=A.data, l=l, q=q)
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

    