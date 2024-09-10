import torch
# import osqp
import numpy as np
from scipy import sparse
import clarabel
from ellipsoids.polytopes_utils import h_rep_minimal, find_interior
import time

class CBF():
    def __init__(self, gsplat, dynamics, alpha, beta, radius):
        # gsplat: GSplat object
        # dynamics: function that returns f, g given x
        # alpha: class K extended function

        self.gsplat = gsplat
        self.dynamics = dynamics
        self.alpha = lambda x: alpha * x
        self.beta = lambda x: beta * x
        self.rel_deg = dynamics.rel_deg
        self.radius = radius

        self.alpha_constant = alpha
        self.beta_constant = beta

        # Create an OSQP object
        # self.prob = osqp.OSQP()

        self.times_solved = 0
        self.solver_success = True

    # TODO: This function assumes relative degree 2, we should make it account for single-integrator dynamics too.
    def get_QP_matrices(self, x, u_des, minimal=True):
        # Computes the A and b matrices for the QP A u <= b
        tnow = time.time()
        h, grad_h, hes_h, info = self.gsplat.query_distance(x[..., :3], radius=self.radius)       # can pass in an optional argument for a radius
        print('Time to query distance:', time.time() - tnow)

        h = h.unsqueeze(-1)
        grad_h = torch.cat((grad_h, torch.zeros(h.shape[0], 3).to(grad_h.device)), dim=-1)
        # add zeros to the right of hes_h
        hes_h = torch.cat((hes_h, torch.zeros(h.shape[0], 3, 3).to(hes_h.device)), dim=2)
        hes_h = torch.cat((hes_h, torch.zeros(h.shape[0], 3, 6).to(hes_h.device)), dim=1)

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

        P = np.eye(3)
        q = -1*u_des.cpu().numpy()

        A = A.cpu().numpy().squeeze()
        l = l.cpu().numpy()

        #We want to solve for a minimal set of constraints in the Polytope
        #Normalize
        norms = np.linalg.norm(A, axis=-1, keepdims=True)
        A = -A / norms
        l = -l / norms.squeeze()

        # Try to find minimal set of polytopes
        tnow = time.time()
        if minimal:

            # We know that the collision-less ellipsoids have CBF constraints that contain the origin. 
            # For those that are in collision, we don't know if the origin is in the polytope and we should
            # avoid trying to solve an optimization problem to find the interior. Because these constraints
            # are relatively few, we can just put them in the QP as is.
            collisionless = (h.cpu().numpy() > 0).squeeze()

            collisionless_A = A[collisionless]
            collisionless_l = l[collisionless]

            collision_A = A[~collisionless]
            collision_l = l[~collisionless]

            print('Is Robot in Collision?: ', np.all(collisionless), 'Number of collisions:', np.sum(~collisionless))

            try:
                try:
                    # If the robot is safe, the origin should be solution (u = -(alpha + beta) v)
                    feasible_pt = -(self.alpha_constant + self.beta_constant) * x[..., 3:6].cpu().numpy()
                    Aminimal, lminimal = h_rep_minimal(collisionless_A, collisionless_l, feasible_pt)
                except:
                    print('The origin is not a feasible point. Resorting to solving Chebyshev center for an interior point.')
                    # Find interior point through Chebyshev center
                    # feasible_pt = find_interior(A, l)
                    # Aminimal, lminimal = h_rep_minimal(A, l, feasible_pt)               
                    raise ValueError('Failed to find an interior point for the minimal polytope.')
                
                print('Reduction in polytope size:', 1 - Aminimal.shape[0] / A.shape[0], 'Final polytope size:', Aminimal.shape[0])
                A, l = np.concatenate([Aminimal, collision_A], axis=0), np.concatenate([lminimal, collision_l], axis=0)
            except:
                print('Failed to compute minimal polytope. Keeping all constraints.')
                pass
        print('Time to compute minimal polytope:', time.time() - tnow)

        return A, l, P, q

    # TODO: We need to make sure that we transform the u_out into the world frame from the ellipsoid frame for ellipsoid-ellipsoid
    def solve_QP(self, x, u_des):
        A, l, P, q = self.get_QP_matrices(x, u_des, minimal=True)

        tnow = time.time()
        u_out, success_flag = self.optimize_QP_clarabel(A, l, P, q)
        print('Time to solve QP:', time.time() - tnow)

        self.solver_success = success_flag

        if success_flag:
            # return the optimal control
            u_out = torch.tensor(u_out).to(device=u_des.device, dtype=torch.float32) 
        else:
            # if not successful, just return the desired control but raise a warning
            print('Solver failed. Returning desired control.')
            u_out = u_des

        #TODO: We might want to try CVXPY as a backup solver if Clarabel fails.

        return u_out

    # This is for OSQP
    def optimize_QP(self, A, l, q):
        udim = A.shape[1]

        # Setup workspace
        P = sparse.eye(udim)
        A = sparse.csc_matrix(A)

        if self.times_solved == 0:
            self.prob.setup(P=P, A=A, l=l, q=q, verbose=False, max_iter=8000)
        else:
            self.prob.update(Ax=A.data, l=l, q=q)
        self.times_solved += 1

        # Solve
        res = self.prob.solve()

        # check if problem is infeasible
        if res.info.status == 'infeasible':
            raise ValueError('OSQP problem is infeasible!')

        # Check solver status
        if res.info.status != 'solved':
            #print number of iters
            print(f"Number of iterations: {res.info.iter}")
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        output = res.x

        return output
    
    # Clarabel is a more robust, faster solver
    def optimize_QP_clarabel(self, A, l, P, q):
        n_constraints = A.shape[0]

        # Setup workspace
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)    

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(P, q, A, l, [clarabel.NonnegativeConeT(n_constraints)], settings)
        sol = solver.solve()

         # Check solver status
        if str(sol.status) != 'Solved':
            print(f"Solver status: {sol.status}")
            print(f"Number of iterations: {sol.iterations}")
            print('Clarabel did not solve the problem!')
            solver_success = False
            solution = None
        else:
            solver_success = True
            solution = sol.x

        return solution, solver_success
