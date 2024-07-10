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
        self.alpha = lambda x: 1e0*5.0 * x
        self.beta = lambda x: 1e0*5.0 * x
        self.rel_deg = dynamics.rel_deg

        # Create an OSQP object
        self.prob = osqp.OSQP()

        self.times_solved = 0

    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        h, grad_h, hes_h = self.gsplat.query_distance(x)       # can pass in an optional argument for a radius

        # print distance h
        # print(f"distance h: {h}")

        # print the min distance
        # print(f"min distance: {torch.min(h)}")

        # lets use a subset of the constraints
        # n = 7000
        # h = h[:n]
        # grad_h = grad_h[:n, :]
        # hes_h = hes_h[:n, :, :]

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
       
        # Extended barrier function
        lfh = torch.matmul(grad_h, f).squeeze()

        lflfh = torch.matmul(f.T, torch.matmul(hes_h, f)).squeeze() + torch.matmul(grad_h, torch.matmul(df, f)).squeeze()
        lglfh = torch.matmul(g.T, torch.matmul(hes_h, f)).squeeze() + torch.matmul(grad_h, torch.matmul(df, g)).squeeze()

        # our full constraint is
        # lflfh + lglfh * u + alpha(lfh) + beta(lfh + alpha(h)) <= 0

        l = -lflfh - self.alpha(lfh) - self.beta(lfh + self.alpha(h.squeeze()))

       
        A = lglfh[None]  # 1 x 6

        P = torch.eye(3).to(grad_h.device)
        qt = torch.tensor(-2*u_des).to(grad_h.device)

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

        # write code here to numerical normalize the constraints

        A = A.cpu().numpy().squeeze()
        l = l.cpu().numpy()
        qt = qt.cpu().numpy()

        # # Normalize the constraints
        # norms = np.linalg.norm(A, axis=1, keepdims=True)
        # A = A / norms
        # l = l / norms.squeeze()

        return A, l

    def solve_QP(self, x, u_des):
        A, l = self.get_QP_matrices(x, u_des, minimal=False)

        q = -1*u_des.cpu().numpy()
        p = self.optimize_QP(A, l, q)       # Need to fill this out

        # return the optimal control
        u = torch.tensor(p).to(device=u_des.device, dtype=torch.float32) 

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

    