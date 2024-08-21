import torch
from ellipsoids.gs_utils import compute_cov

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

def compute_ellipsoid_gradients(R, D, R_robot, D_robot, mu_A, mu_B):
    """
    Compute the gradients of the ellipsoid intersection test.

    Args:
        R (torch.Tensor): The quaternions of the ellipsoid A. Shape (batch_dim, 4).
        D (torch.Tensor): The axes lengths of the ellipsoid A. Shape (batch_dim, state_dim).
        R_robot (torch.Tensor): The quaternions of the ellipsoid B. Shape (batch_dim, 4).
        D_robot (torch.Tensor): The axes lengths of the ellipsoid B. Shape (batch_dim, state_dim).
        mu_A (torch.Tensor): The mean of ellipsoid A. Shape (batch_dim, state_dim).
        mu_B (torch.Tensor): The mean of ellipsoid B. Shape (batch_dim, state_dim).

    Returns:
        K_j (torch.Tensor): The maximum K value for each batch. Shape (batch_dim,).
        grads (torch.Tensor): The gradients of the ellipsoid intersection test. Shape (batch_dim, state_dim).

    """
    batch = R.shape[0]

    # Compute covariance matrices
    Sigma_A = compute_cov(R, D)
    Sigma_B = 0.01*torch.eye(3).to(device) #compute_cov(R_robot.reshape(1, -1), D_robot.reshape(1, -1))

    # Compute lambdas, Phi, and v_squared
    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(Sigma_A, Sigma_B.squeeze(), mu_A, mu_B.reshape(1, -1))

    # Compute KK
    KK = ellipsoid_K_function(lambdas, v_squared, 1.)

    # Compute K_j and inds
    K_j, inds = torch.max(KK, dim=-1)

    # Compute s_max
    ss = torch.linspace(0., 1., 100, device=device)[1:-1]
    s_max = ss[inds]

    # Compute S_j
    S_j_flat = (s_max*(1-s_max))[..., None] / (1. + s_max[..., None] * (lambdas - 1.))
    S_j = torch.diag_embed(S_j_flat)

    # Compute A_j
    A_j = torch.bmm(Phi, torch.bmm(S_j, Phi.transpose(1, 2)))

    # Compute grads
    grads = torch.bmm((mu_A - mu_B[None, :]).reshape(batch, 1, -1), A_j).squeeze()

    print(K_j.min())
    raise

    return K_j-1., -grads, A_j