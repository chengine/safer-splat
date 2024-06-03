import numpy as np
from matplotlib.patches import Ellipse, Polygon
import torch
import time
from polytope import extreme, cheby_ball, bounding_box
import polytope
from cvxopt import matrix, solvers
import scipy
import cvxpy as cvx
from scipy.optimize import linprog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generalized_eigen(A, B):
    # IMPORTANT!!! Assuming B is not batched (statedim x statedim), A is batched (batchdim x statedim x statedim)
    batch_dim = A.shape[0]
    state_dim = B.shape[0]

    # see NR section 11.0.5
    # L is a lower triangular matrix from the Cholesky decomposition of B
    L,_ = torch.linalg.cholesky_ex(B)

    L = L.reshape(-1, state_dim, state_dim).expand(batch_dim, -1, -1)

    # solve Y * L^T = A by a workaround
    # if https://github.com/tensorflow/tensorflow/issues/55371 is solved then this can be simplified
    Y = torch.transpose(torch.linalg.solve_triangular(L, torch.transpose(A, 1, 2), upper=False), 1, 2)

    # solve L * C = Y
    C = torch.linalg.solve_triangular(L, Y, upper=False)
    # solve the equivalent eigenvalue problem

    e, v_ = torch.linalg.eigh(C)

    # solve L^T * x = v, where x is the eigenvectors of the original problem
    v = torch.linalg.solve_triangular(torch.transpose(L, 1, 2), v_, upper=True)
    # # normalize the eigenvectors
    return e, v

def ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau):
    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    KK = ellipsoid_K_function(lambdas, v_squared, tau)      # batchdim x Nsamples
    return ~torch.all(torch.any(KK > 1., dim=-1))

def ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B):
    lambdas, Phi = generalized_eigen(Sigma_A, Sigma_B) # eigh(Sigma_A, b=Sigma_B)
    v_squared = (torch.bmm(Phi.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, Phi, v_squared

def ellipsoid_K_function(lambdas, v_squared, tau):
    batchdim = lambdas.shape[0]
    ss = torch.linspace(0., 1., 100, device=device)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*torch.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(1.+ss*(lambdas.reshape(batchdim, 1, -1)-1.))), dim=2)

def gs_sphere_intersection_test(R, D, kappa, mu_A, mu_B, tau, return_raw=False):
    # tnow = time.time()
    lambdas, v_squared = gs_sphere_intersection_test_helper(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    # print('helper:' , time.time() - tnow)
    # tnow = time.time()
    KK = gs_K_function(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    # print('function eval:' , time.time() - tnow)

    if return_raw:
        test_result = torch.any(KK > 1., dim=-1)
    else:
        # tnow = time.time()
        test_result = ~torch.all(torch.any(KK > 1., dim=-1))
        # print('boolean:' , time.time() - tnow)

    return test_result

def gs_sphere_intersection_test_helper(R, D, mu_A, mu_B):
    lambdas, v_squared = D, (torch.bmm(R.transpose(1, 2), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def gs_K_function(lambdas, v_squared, kappa, tau):
    batchdim = lambdas.shape[0]
    ss = torch.linspace(0., 1., 100, device=device)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*torch.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(kappa + ss*(lambdas.reshape(batchdim, 1, -1) - kappa))), dim=2)

def gs_sphere_intersection_eval(R, D, kappa, mu_A, mu_B, tau):
    lambdas, v_squared = gs_sphere_intersection_test_helper(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    KK = gs_K_function(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    K = torch.max(KK, dim=-1)
    return K

def gs_sphere_intersection_test_np(R, D, kappa, mu_A, mu_B, tau):
    tnow = time.time()
    lambdas, v_squared = gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B)  # (batchdim x statedim), (batchdim x statedim x statedim), (batchdim x statedim)
    print('helper:' , time.time() - tnow)
    tnow = time.time()
    KK = gs_K_function_np(lambdas, v_squared, kappa, tau)      # batchdim x Nsamples
    print('function eval:' , time.time() - tnow)

    tnow = time.time()
    test_result = ~np.all(np.any(KK > 1., axis=-1))
    print('boolean:' , time.time() - tnow)

    return test_result

def gs_sphere_intersection_test_helper_np(R, D, mu_A, mu_B):
    lambdas, v_squared = D, (np.matmul(np.transpose(R, (0, 2, 1)), (mu_A - mu_B)[..., None])).squeeze() ** 2
    return lambdas, v_squared

def gs_K_function_np(lambdas, v_squared, kappa, tau):
    batchdim = lambdas.shape[0]
    ss = np.linspace(0., 1., 100)[1:-1].reshape(1, -1, 1)
    return (1./tau**2)*np.sum(v_squared.reshape(batchdim, 1, -1)*((ss*(1.-ss))/(kappa + ss*(lambdas.reshape(batchdim, 1, -1) - kappa))), axis=2)

def compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau):

    batch = R.shape[0]
    dim = R.shape[-1]

    evals = gs_sphere_intersection_eval(R, D, kappa, mu_A, test_pt, tau)

    K_j = evals[0]
    inds = evals[1]

    ss = torch.linspace(0., 1., 100, device=device)[1:-1]
    s_max = ss[inds]

    lambdas = D

    S_j_flat = (s_max*(1-s_max))[..., None] / (kappa + s_max[..., None] * (lambdas - kappa))

    S_j = torch.diag_embed(S_j_flat)
    A_j = torch.bmm(R, torch.bmm(S_j, R.transpose(1, 2)))

    delta_j = test_pt - mu_A

    A = -torch.bmm(delta_j.reshape(batch, 1, -1), A_j).squeeze()
    b = -torch.sqrt(K_j) + torch.sum(A*mu_A, dim=-1)

    proj_points = mu_A + delta_j / torch.sqrt(K_j)[..., None]

    return A.cpu().numpy().reshape(-1, dim), b.cpu().numpy().reshape(-1, 1), proj_points.cpu().numpy()

def compute_polytope(R, D, kappa, mu_A, test_pt, tau, A_bound, b_bound):
    # Find safe polytope in A <= b form
    A, b, _ = compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pt, tau)

    # A, b = A.cpu().numpy(), b.cpu().numpy()

    dim = mu_A.shape[-1]

    # Add in the bounding poly constraints
    A = np.concatenate([A.reshape(-1, dim), A_bound.reshape(-1, dim)], axis=0)
    b = np.concatenate([b.reshape(-1, 1), b_bound.reshape(-1, 1)], axis=0)

    # poly = polytope.Polytope(A, b)
    # tnow = time.time()
    # reduced_poly = polytope.reduce(poly)
    # print('Time to reduce polytope: ', time.time() - tnow)
    return A, b

def h_rep_minimal(A, b, pt):
    halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
    hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
    minimal_Ab = halfspaces[hs.dual_vertices]
    # qhull_pts = hs.intersections

    # convex_hull = scipy.spatial.ConvexHull(qhull_pts, incremental=False, qhull_options=None)
    # minimal_Ab = convex_hull.equations

    minimal_A = minimal_Ab[:, :-1]
    minimal_b = -minimal_Ab[:, -1]

    return minimal_A, minimal_b

def find_interior(A, b):
    # by way of Chebyshev center
    norm_vector = np.reshape(np.linalg.norm(A, axis=1),(A.shape[0], 1))
    c = np.zeros(A.shape[1]+1)
    c[-1] = -1
    A = np.hstack((A, norm_vector))

    # print(c.shape, A.shape, b.shape)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))

    return res.x[:-1]

def sample_sphere(N_samples):
    # Samples the unit sphere uniformly
    # reference: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    i = np.arange(N_samples)
    y = 1. - (i / (N_samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    theta = phi * i  # golden angle increment

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return np.stack([x, y, z], axis=-1)

def sphere_to_poly(N_samples, radius, center):

    sphere_v_rep = sample_sphere(N_samples)

    # multiply by radius
    sphere_v_rep *= radius

    # then shift by the center
    sphere_v_rep = sphere_v_rep + center[None,...]

    poly = polytope.qhull(sphere_v_rep)

    return poly, sphere_v_rep

def circle_to_poly(N_samples, radius, center):

    t = np.linspace(0., 2*np.pi, N_samples)
    circle_v_rep = np.stack([np.cos(t), np.sin(t)], axis=-1)
    circle_v_rep = radius * circle_v_rep + center[None,...]

    poly = polytope.qhull(circle_v_rep)

    return poly

def test_connected_union_polys(As, bs):
    # polys: list of polytopes

    # Test if the union of polytopes is a connected region

    As_1 = As[:-1]
    As_2 = As[1:]

    bs_1 = bs[:-1]
    bs_2 = bs[1:]

    adjacents = []
    for A1, A2, b1, b2 in zip(As_1, As_2, bs_1, bs_2):
        adjacent = polytope.is_adjacent(polytope.Polytope(A1, b1), polytope.Polytope(A2, b2))  # returns boolean
        adjacents.append(adjacent)

    adjacents = np.array(adjacents)
    # print(adjacents)
    return np.all(adjacents)

def test_connected_union_polys_approximate(As, bs, points):
    As_1 = As[:-1]
    As_2 = As[1:]

    bs_1 = bs[:-1]
    bs_2 = bs[1:]

    pts_1 = points[:-1]
    pts_2 = points[1:]

    adjacents = []
    for A1, A2, b1, b2, p1, p2 in zip(As_1, As_2, bs_1, bs_2, pts_1, pts_2):
        # Use only points that are interior
        keep1 = np.all((A1 @ p1.T - b1[..., None]) <= 1e-2, axis=0)
        keep2 = np.all((A2 @ p2.T - b2[..., None]) <= 1e-2, axis=0)

        keep_pts1 = p1[keep1]
        keep_pts2 = p2[keep2]

        # Test if valid points in 1 satisfy polytope 2
        poly1_in_2 = np.any(np.all(A2 @ keep_pts1.T - b2[..., None] <= 1e-2, axis=0))
        poly2_in_1 = np.any(np.all(A1 @ keep_pts2.T - b1[..., None] <= 1e-2, axis=0))

        # print(poly2_in_1)
        adjacents.append(poly1_in_2 or poly2_in_1)

    adjacents = np.array(adjacents)
    # print(adjacents)

    return np.all(adjacents)

def test_connected_union_polys_LP(As, bs):
    N = len(As)-1
    dim = As[0].shape[-1]

    As1 = As[:-1]
    As2 = As[1:]
    bs1 = bs[:-1]
    bs2 = bs[1:]

    A_intersection = [np.concatenate([A1, A2], axis=0) for (A1, A2) in zip(As1, As2)]
    b_intersection = [np.concatenate([b1, b2], axis=0) for (b1, b2) in zip(bs1, bs2)]
    A, b = polytopes_to_matrix(A_intersection, b_intersection)

    c = matrix(np.zeros(A.shape[-1]))
    A = matrix(A)
    b = matrix(b)
    sol=solvers.lp(c,A,b, solver='glpk')

    # test_pts = cvx.Variable((N, dim))

    # obj = cvx.Minimize(0.)

    # constraints = [A @ cvx.reshape(test_pts, N*dim, order='C') <= b]

    # prob = cvx.Problem(obj, constraints)
    # prob.solve(solver='GLPK')
    # prob.solve()

    # print(sol['x'])
    if sol['x'] is not None:
        return True
    else:
        return False

def polytopes_to_matrix(As, bs):
    A_sparse = scipy.linalg.block_diag(*As)
    b_sparse = np.concatenate(bs)

    return A_sparse, b_sparse

def check_and_project(A, b, point):
    # Check if Ax <= b

    criteria = A @ point - b
    is_valid = np.all(criteria < 0)

    if is_valid:
        return point
    else:
        # project point to nearest facet
        pt = cvx.Variable(3)
        obj = cvx.Minimize(cvx.norm(pt - point))
        constraints = [A @ pt <= b]
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver='CLARABEL')
        return pt.value

def plot_ellipse(mu, Sigma, n_std_tau, ax, **kwargs):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb

    ee, V = torch.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = torch.arctan(v_big[1] / v_big[0]) * 180. / torch.tensor(np.pi)

    long_length = n_std_tau * 2. * torch.sqrt(e_big)
    short_length = n_std_tau * 2. * torch.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'none'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)

def get_patch(poly1, color="green"):
    """Takes a Polytope and returns a Matplotlib Patch Polytope 
    that can be added to a plot
    
    Example::

    > # Plot Polytope objects poly1 and poly2 in the same plot
    > import matplotlib.pyplot as plt
    > fig = plt.figure()
    > ax = fig.add_subplot(111)
    > p1 = get_patch(poly1, color="blue")
    > p2 = get_patch(poly2, color="yellow")
    > ax.add_patch(p1)
    > ax.add_patch(p2)
    > ax.set_xlim(xl, xu) # Optional: set axis max/min
    > ax.set_ylim(yl, yu) 
    > plt.show()
    """
    V = extreme(poly1)
    rc,xc = cheby_ball(poly1)
    x = V[:,1] - xc[1]
    y = V[:,0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x/mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2*(y < 0)
    angle = angle*corr
    ind = np.argsort(angle) 

    patch = Polygon(V[ind,:], True, color=color, alpha=0.1)
    return patch

def plot_polytope(poly1, ax, color='green'):
    """Plots a 2D polytope or a region using matplotlib.
    
    Input:
    - `poly1`: Polytope or Region
    """
    if len(poly1) == 0:

        poly = get_patch(poly1, color)
        l,u = bounding_box(poly1)
        ax.add_patch(poly)        

    else:
        l,u = bounding_box(poly1, color)

        for poly2 in poly1.list_poly:
            poly = get_patch(poly2, color=np.random.rand(3))
            ax.add_patch(poly)
    return ax

def plot_halfplanes(A, b, lower, upper, ax):
    # In A x <= b form in 2D

    t = np.linspace(lower, upper, endpoint=True)

    y = (-A[:, 0][:, None]*t[None,...] + b[:, None]) / A[:, 1][:, None]

    for y_ in y:
        ax.plot(t, y_, linestyle='dotted')

    return ax