import torch
import numpy as np
import torch.linalg as la
import cvxpy as cp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
# http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
# https://minillinim.github.io/GroopM/dev_docs/groopm.ellipsoid-pysrc.html
def mvee(points):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    # dot = lambda foo, bar : torch.tensordot(foo, bar, dims=1)
    N, d = points.shape
    Q = torch.vstack((points.T, torch.ones(N, device=device))) # Q.shape = (d+1, N)
    # err = tol+1.0
    u = torch.ones(N, device = device)/N # u.shape = (N,)
    arange = torch.arange(N, device=device)
    # while err > tol:
    for i in range(10):
        X = (Q * u[None, :]) @ Q.T # shapes: (((d+1, N), (N, N)) , (N, d+1)) = (d+1, d+1)
        M = torch.sum( (Q.T @ la.inv(X))*Q.T, dim = 1) # (((N, d+1), (d+1, d+1)), (d+1, N)) = (N, N); M.shape = (N,)
        mmax, mind = torch.max(M, 0)
        step_size = (mmax-d-1.0)/((d+1)*(mmax-1.0))
        u = (1-step_size)*u
        u += torch.where(arange == mind, step_size, 0)
        # new_u[mind] += step_size
        # err = la.norm(new_u-u)
        # u = new_u
    c = u @ points
    A = (points.T * u[None, :]) @ points - torch.outer(c, c)

    # Projection step

    val = torch.einsum('bi, ij, bj-> b', (points-c[None,:], la.inv(A)/d, points-c[None,:])).max()
    # val0, ind0 = torch.max(val[:N//2], 0)
    # val1, ind1 = torch.max(val[N//2:], 0)
    # topk_points = torch.index_select(points, 0, torch.stack([ind0, ind1]))
    # a = topk_points[0] - c
    # b = -2*torch.einsum('bi, ij, bj-> b', (topk_points-c[None,:], la.inv(A)/d, a.expand(topk_points.shape)))
    # t = (val0 - val1) / (b[1]-b[0])

    # c = c + t*a
    # val = torch.einsum('bi, ij, bj-> b', (topk_points-c[None,:], la.inv(A)/d, topk_points-c[None,:])).max()
    # val = 1.
    return A*d*val, c

def mvee_batched(points, tol=0.01):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    points: (B x N x d)
    """
    # dot = lambda foo, bar : torch.tensordot(foo, bar, dims=1)
    B, N, d = points.shape
    Q = torch.cat((points, torch.ones_like(points[..., 0:1], device=device)), dim=-1) # Q.shape = (B, N, d+1)
    Q = Q.permute(0, 2, 1) # Q.shape = (B, d+1, N)
    err = tol+1.0
    u = torch.ones((B, N), device = device)/N # u.shape = (N,)
    # while err > tol:
    # for i in range(50):
    X = torch.bmm(Q * u[..., None, :], Q.permute(0, 2, 1)) # shapes: (((d+1, N), (N, N)) , (N, d+1)) = (B, d+1, d+1)

    M = torch.einsum('bij, bjl, bli -> bi', Q.permute(0, 2, 1), la.inv(X), Q)

    mmax, mind = torch.max(M, 1)    # (B)
    step_size = (mmax-d-1.0)/((d+1)*(mmax-1.0)) # B
    new_u = (1-step_size)[..., None] * u        # B x N
    indices = torch.stack([torch.arange(B, device=device), mind], dim=-1)
    new_u[indices[:, 0], indices[:, 1]] += step_size
    err = la.norm(new_u-u, dim=-1).max()
    u = new_u       # B x N
        # count += 1
    c = torch.bmm(u[..., None, :], points).squeeze()
    A = torch.bmm(points.permute(0, 2, 1) * u[..., None, :], points) - torch.einsum('bi,bj->bij', (c, c))
    return A*d, c

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