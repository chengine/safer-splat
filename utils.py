#%%
import numpy as np
# import torch
import cvxpy as cp
import scipy
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time
from downsampling.utils import mvee
from ellipsoids.utils import fibonacci_ellipsoid
import torch

device = torch.device('cuda')

def outer_ellipsoid(A1, b1, A2, b2):

    As = [np.linalg.inv(A1), np.linalg.inv(A2)]
    bs = [(-np.linalg.inv(A1)@b1).reshape(-1, 1), (-np.linalg.inv(A2)@b2).reshape(-1, 1)]
    cs = [(b1.T@np.linalg.inv(A1)@b1 - 1).reshape(-1, 1), (b2.T@np.linalg.inv(A2)@b2 - 1).reshape(-1, 1)]

    n = A1.shape[0]
    m = 2

    Asqr = cp.Variable((n,n), symmetric=True)
    btilde = cp.Variable((n,1))
    t = cp.Variable(m, pos=True)
    obj = cp.Maximize( cp.log_det( Asqr ) )
    constraints = []
    for i in range(m):
        constraints += [
            cp.vstack([cp.hstack([-(Asqr - t[i] * As[i]), -(btilde - t[i]*bs[i]), np.zeros((n,n))]),
            cp.hstack([-(btilde - t[i]*bs[i]).T, -(- 1 - t[i]*cs[i]), -btilde.T]),
            cp.hstack([np.zeros((n,n)), -btilde, Asqr])
            ]) >> 0
            ]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    A = np.linalg.inv(Asqr.value)
    b = -A@btilde.value

    return A, b

# Example usage
A1 = np.array([[1, 0], [0, 2]])  # Covariance matrix of inner ellipsoid 1
b1 = np.array([1, 2])  # Center of inner ellipsoid 1
A2 = np.array([[2, 0], [0, 1]])  # Covariance matrix of inner ellipsoid 2
b2 = np.array([-1, -2])  # Center of inner ellipsoid 2

tnow = time.time()
covariance, center = outer_ellipsoid(A1, b1, A2, b2)
# print('Elapsed', time.time() - tnow)
# print("Center:", center)
print("Cov:", covariance)

scale1, rot1 = np.linalg.eigh(A1)
scale2, rot2 = np.linalg.eigh(A1)

points = fibonacci_ellipsoid(torch.tensor(np.stack([b1, b2], axis=0), device=device), torch.tensor(np.stack([rot1, rot2], axis=0)), torch.tensor(np.stack([scale1, scale2], axis=0)), 0., n=10)
points = torch.stack([points[:-1], points[1:]], dim=1)
points = torch.flatten(points, start_dim=1, end_dim=2)
mvee
#%%

# Plotting function for ellipsoids
def plot_ellipse(mu, Sigma, n_std_tau, ax, **kwargs):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb

    ee, V = np.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = np.arctan2(v_big[1] , v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * np.sqrt(e_big)
    short_length = n_std_tau * 2. * np.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'grey'
    if not ('edgecolor' in kwargs):
        kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)

# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(15, 15))

# Plot the outer ellipsoid
kwargs = {
    'facecolor': 'none',
    'edgecolor': 'g',
    'linewidth': 2

}

plot_ellipse(center, covariance, 1., ax, **kwargs)

# Plot the inner ellipsoids
plot_ellipse(b1, A1, 1, ax)
plot_ellipse(b2, A2, 1, ax)

# Set the aspect ratio to equal
ax.set_aspect('equal')

# Set the limits of the plot
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

# Show the plot
# plt.show()
# %%
