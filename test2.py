#%%
import torch
import torch.linalg as la
import numpy as np
import time
import matplotlib.pyplot as plt
from downsampling.utils import mvee, mvee_batched, outer_ellipsoid
from ellipsoids.utils import fibonacci_ellipsoid, rot_z
from ellipsoids.plots import plot_ellipse
from torch.profiler import profile, ProfilerActivity, record_function

device = torch.device('cuda')

torch.manual_seed(0)

n = 600000
# dimension
n_dim = 2

thetas = torch.rand(n).cuda()
mu_A = torch.rand(n, n_dim).cuda()
rot_A = rot_z(thetas, n_dim=n_dim)
scale_A = 5e-2 + torch.rand(n, n_dim).cuda()
sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))
# scale_A = scale_A**2

points = fibonacci_ellipsoid(mu_A, rot_A, scale_A, 0., n=100)
points = torch.stack([points[:-1], points[1:]], dim=1)
points = torch.flatten(points, start_dim=1, end_dim=2)
func = torch.vmap(mvee, in_dims=0)

#%%
tnow = time.time()
torch.cuda.synchronize()
# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with record_function("model_inference"):
covariance, center = func(points)
#print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
torch.cuda.synchronize()
print('Elapsed', time.time() - tnow)

# tnow = time.time()
# torch.cuda.synchronize()
# covariance, center = mvee_batched(points)
# torch.cuda.synchronize()
# print('Elapsed', time.time() - tnow)
# for i in range(10):
#     with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with record_function("model_inference"):
#             tnow = time.time()
#             torch.cuda.synchronize()
#             covariance, center = mvee_batched(points)
#             torch.cuda.synchronize()

#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#     print('Elapsed', time.time() - tnow)

# covariance, center = mvee(points[0])
# print("Center:", center)
# print("Cov:", covariance)

#%%

for i in range(10):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(15, 15))

    # Plot the outer ellipsoid
    kwargs = {
        'facecolor': 'none',
        'edgecolor': 'g',
        'linewidth': 2

    }

    # ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), s=1)

    # plot_ellipse(center[:2], covariance[:2, :2], 1., ax, **kwargs)
    plot_ellipse(center[i, :2], covariance[i, :2, :2], 1., ax, **kwargs)

    # Plot the inner ellipsoids
    for j, (mu, sigma) in enumerate(zip(mu_A[i:i+2], Sigma_A[i:i+2])):
        plot_ellipse(mu, sigma, 1, ax)

        if j == 1:
            break

    A, b = outer_ellipsoid(Sigma_A[i:i+2][0].cpu().numpy(), mu_A[i:i+2][0].cpu().numpy(), Sigma_A[i:i+2][1].cpu().numpy(), mu_A[i:i+2][1].cpu().numpy())
    A = torch.tensor(A)
    b = torch.tensor(b)
    plot_ellipse(b, A, 1., ax, facecolor='none', edgecolor='r', linewidth=2)

    # Set the aspect ratio to equal
    ax.set_aspect('equal')

    # Set the limits of the plot
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    plt.savefig('ellipsoids2.png')
# %%
