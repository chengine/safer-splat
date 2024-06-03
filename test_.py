#%%
import os
import numpy as np
import time
from pykdtree.kdtree import KDTree 
import collections
import json
import torch
import torch.linalg as la
import matplotlib.pyplot as plt
from downsampling.utils import mvee, mvee_batched, outer_ellipsoid
from ellipsoids.utils import fibonacci_ellipsoid, rot_z, create_gs_mesh
from ellipsoids.plots import plot_ellipse
from ellipsoids.gs_utils import quaternion_to_rotation_matrix
import open3d as o3d

device = torch.device('cuda')

with open('flightroom_gaussians_sparse_deep.json', 'r') as f:
    data = json.load(f)

#%% 
means = np.array(data['means']).astype(np.float32)
rots = torch.tensor(data['rotations']).to(dtype=torch.float32, device=device)
opacities = torch.sigmoid(torch.tensor(data['opacities']).to(dtype=torch.float32, device=device))
colors = torch.tensor(data['colors']).to(dtype=torch.float32, device=device)
scales = torch.exp(torch.tensor(data['scalings']).to(dtype=torch.float32, device=device))

scene = create_gs_mesh(means, quaternion_to_rotation_matrix(rots).cpu().numpy(), scales.cpu().numpy(), colors.cpu().numpy(), res=4, transform=None, scale=None)
o3d.io.write_triangle_mesh('flightroom_gaussians_sparse_deep.obj', scene)

# scene = np.random.rand(10000, 3).astype(np.float32)

tnow = time.time()
kept = 1.
for i in range(1):
    scene_tree = KDTree(means)
    dist, idx = scene_tree.query(means, k=2)# , distance_upper_bound=0.01)

    # pairs = []
    # for source, target in np.array(idx):
    #     away = np.where(idx[:, -1] == source)[0]
    #     if away.size:
    #         pairi = np.concatenate([source.reshape(-1,), away])
    #     else:
    #         pairi = source

    #     pairs.append(pairi)

    ind = idx[:, -1]
    inds = ind[ind]
    keep = (idx[:, 0] != inds)
    keep_pairs = idx[keep]

    keep_cycles = ~keep
    cycles_pairs = idx[keep_cycles]
    cycles_pairs = cycles_pairs[cycles_pairs[:, 0] < cycles_pairs[:, 1]]

    # total_pairs = keep_pairs
    total_pairs = np.concatenate([keep_pairs, cycles_pairs], axis=0)

    kept *= len(total_pairs) / len(idx)
    print('Kept:', kept)

    total_pairs = torch.from_numpy(total_pairs.astype(np.int64))

    means = torch.from_numpy(means).to(device)
    means1 = means[total_pairs[:, 0]]
    means2 = means[total_pairs[:, 1]]
    rots1 = rots[total_pairs[:, 0]]
    rots2 = rots[total_pairs[:, 1]]
    scales1 = scales[total_pairs[:, 0]]
    scales2 = scales[total_pairs[:, 1]]
    colors1 = colors[total_pairs[:, 0]]
    colors2 = colors[total_pairs[:, 1]]
    opacities1 = opacities[total_pairs[:, 0]]
    opacities2 = opacities[total_pairs[:, 1]]

# vals, inverse, count = np.unique(idx[:, -1],
#                                  return_inverse=True,
#                                  return_counts=True)

# count = collections.Counter(inverse)
# ind = idx[:, -1]

# keep = ind[ind]

# keep_pairs = idx[np.unique(keep)]
# pairs = np.delete(idx, ind, axis=0)

# total_pairs = np.vstack([pairs, keep_pairs])

# np.unique(idx[:, -1], return_index=True)

# downsampled_idx = np.delete(idx[:, -1], ind)
# downsampled = np.delete(np.arange(scene.shape[0]), ind)

# downsampled_scene = scene[downsampled]
# downsampled_idx = idx[downsampled]
print('Elapsed', time.time() - tnow)

total = total_pairs.reshape(-1)
print(np.unique(total).shape)

print('Sanity Check:', len(np.unique(total))/ len(idx))
# %%

points1 = fibonacci_ellipsoid(means1, quaternion_to_rotation_matrix(rots1), scales1, 0., n=100)
points2 = fibonacci_ellipsoid(means2, quaternion_to_rotation_matrix(rots2), scales2, 0., n=100)

points = torch.stack([points1, points2], dim=1)
points = torch.flatten(points, start_dim=1, end_dim=2)
func = torch.vmap(mvee, in_dims=0)

#%%
tnow = time.time()
torch.cuda.synchronize()
# with profile(activities=[
#         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with record_function("model_inference"):
covariance, center = func(points)
# %%

means = center
eigs, eigvecs = torch.linalg.eigh(covariance)

scales = eigs.sqrt()
rotations = eigvecs
colors = (colors1 + colors2) / 2
opacities = (opacities1 + opacities2) / 2

scene = create_gs_mesh(means.cpu().numpy(), rotations.cpu().numpy(), scales.cpu().numpy(), colors.cpu().numpy(), res=4, transform=None, scale=None)
o3d.io.write_triangle_mesh('flightroom_gaussians_sparse_deep_downsampled.obj', scene)

data = {
    'means': means.cpu().numpy().tolist(),
    'rotations': rotations.cpu().numpy().tolist(),
    'opacities': opacities.cpu().numpy().tolist(),
    'colors': colors.cpu().numpy().tolist(),
    'scalings': scales.cpu().numpy().tolist()
}

with open('flightroom_gaussians_sparse_deep_downsample.json', 'w') as f:
    json.dump(data, f, indent=4)
# %%
