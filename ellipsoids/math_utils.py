import numpy as np
from matplotlib.patches import Ellipse, Polygon
import torch
import time
import scipy
import cvxpy as cvx
from scipy.optimize import linprog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
