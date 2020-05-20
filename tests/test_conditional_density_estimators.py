""" Test conditional normalizing flows."""

import numpy as np
import torch
import torch_nf.density_estimator as de
from torch_nf.conditional_density_estimator import ConditionalDensityEstimator
from torch_nf.bijectors import Bijector, RealNVP, MAF, BatchNorm, ToSimplex, Affine
from pytest import raises

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def test_ConditionalDensityEstimator():
    D = 4
    nf = de.NormFlow(D, True, 'coupling', 1, 2, 20, None)
    D_x = 10
    hidden_layers = [50, 100]
    cde = ConditionalDensityEstimator(nf, D_x, hidden_layers)

    M = 20
    N = 50
    x = torch.tensor(np.random.normal(0., 1., (M, cde.D_x))).float()
    z, log_q_z = cde(x, N=N)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == M and z.shape[1] == N)

    log_q_z_inv = cde.log_prob(z, x)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    D = 4
    nf = de.NormFlow(D, True, 'AR', 1, 2, 20, None)
    D_x = 10
    hidden_layers = [50, 50]
    cde = ConditionalDensityEstimator(nf, D_x, hidden_layers)

    M = 20
    N = 50
    x = torch.tensor(np.random.normal(0., 1., (M, cde.D_x))).float()
    z, log_q_z = cde(x, N=N)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == M and z.shape[1] == N)

    log_q_z_inv = cde.log_prob(z, x)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    D = 4
    K = 3
    mog = de.MoG(D, True, K)
    D_x = 10
    hidden_layers = [50, 50]
    cde = ConditionalDensityEstimator(mog, D_x, hidden_layers)

    M = 20
    N = 50
    x = torch.tensor(np.random.normal(0., 1., (M, cde.D_x))).float()
    z, log_q_z = cde(x, N=N)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z is None)

    return None

if __name__ == "__main__":
    test_ConditionalDensityEstimator()
