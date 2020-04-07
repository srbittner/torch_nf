""" Test conditional normalizing flows."""

import numpy as np
import torch
from torch_nf.conditional_nf import NormFlow, ConditionedNormFlow
from torch_nf.bijectors import Bijector, RealNVP, MAF, BatchNorm, ToSimplex, Affine
from pytest import raises

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def test_NormFlow():
    """Test NormFlow class initialization."""

    D = 4
    arch_type = "coupling"
    conditioner = False
    num_stages = 1
    num_layers = 2
    num_units = 30
    support_layer = None
    nf = NormFlow(
        D, arch_type, conditioner, num_stages, num_layers, num_units, support_layer
    )
    assert nf.D == D
    assert nf.arch_type == arch_type
    assert nf.num_stages == num_stages
    assert nf.num_layers == num_layers
    assert nf.num_units == num_units
    assert nf.support_layer == support_layer

    nf = NormFlow(
        D, arch_type, conditioner, num_stages, num_layers, 10, ToSimplex(D)
    )

    assert nf.num_units == 15
    assert issubclass(type(nf.support_layer), Bijector)

    with raises(TypeError):
        nf = NormFlow('foo', 'coupling', False, 1, 2, 20, None)
    with raises(ValueError):
        nf = NormFlow(-1, 'coupling', False, 1, 2, 20, None)

    with raises(TypeError):
        nf = NormFlow(4, 1, False, 1, 2, 20, None)
    with raises(ValueError):
        nf = NormFlow(4, 'foo', False, 1, 2, 20, None)

    with raises(TypeError):
        nf = NormFlow(4, 'coupling', 1, 1, 2, 20, None)

    with raises(TypeError):
        nf = NormFlow(4, 'coupling', False, 'foo', 2, 20, None)
    with raises(ValueError):
        nf = NormFlow(4, 'coupling', False, -1, 2, 20, None)

    with raises(TypeError):
        nf = NormFlow(4, 'coupling', False, 1, 'foo', 20, None)
    with raises(ValueError):
        nf = NormFlow(4, 'coupling', False, 1, -1, 20, None)

    with raises(TypeError):
        nf = NormFlow(4, 'coupling', False, 1, 2, 'foo', None)
    with raises(ValueError):
        nf = NormFlow(4, 'coupling', False, 1, 2, -1, None)

    with raises(TypeError):
        nf = NormFlow(4, 'coupling', False, 1, 2, 20, 'foo')

    nf = NormFlow(D, 'coupling', False, 1, 2, 20, None)

    N = 10
    z, log_q_z = nf(N)
    assert(z.shape[0] == 1 and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == 1 and log_q_z.shape[1] == N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)
    
    nf = NormFlow(D, 'coupling', True, 2, 2, 20, ToSimplex(D))
    assert(issubclass(type(nf.bijectors[0]), RealNVP))
    assert(issubclass(type(nf.bijectors[1]), BatchNorm))
    assert(issubclass(type(nf.bijectors[2]), RealNVP))
    assert(issubclass(type(nf.bijectors[3]), BatchNorm))
    assert(issubclass(type(nf.bijectors[4]), Affine))
    assert(issubclass(type(nf.bijectors[5]), RealNVP))
    assert(issubclass(type(nf.bijectors[6]), BatchNorm))
    assert(issubclass(type(nf.bijectors[7]), RealNVP))
    assert(issubclass(type(nf.bijectors[8]), BatchNorm))
    assert(issubclass(type(nf.bijectors[9]), Affine))
    assert(issubclass(type(nf.bijectors[10]), ToSimplex))

    nf = NormFlow(D, 'coupling', False, 2, 2, 20)
    assert(issubclass(type(nf.bijectors[0]), RealNVP))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    nf = NormFlow(D, 'autoregressive', False, num_layers=2, num_units=20)
    assert(issubclass(type(nf.bijectors[0]), MAF))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    return None

def test_ConditionedNormFlow():
    D = 4
    nf = NormFlow(D, 'coupling', True, 1, 2, 20, None)
    D_x = 10
    hidden_layers = [50, 100]
    cnf = ConditionedNormFlow(nf, D_x, hidden_layers)

    M = 20
    N = 50
    eta = torch.tensor(np.random.normal(0., 1., (M, cnf.D_x))).float()
    z, log_q_z = cnf(eta, N=N)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == M and z.shape[1] == N)

    log_q_z_inv = cnf.log_prob(z, eta)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    D = 4
    nf = NormFlow(D, 'autoregressive', True, 1, 2, 20, None)
    D_x = 10
    hidden_layers = [50, 50]
    cnf = ConditionedNormFlow(nf, D_x, hidden_layers)

    M = 20
    N = 50
    eta = torch.tensor(np.random.normal(0., 1., (M, cnf.D_x))).float()
    z, log_q_z = cnf(eta, N=N)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == M and z.shape[1] == N)

    log_q_z_inv = cnf.log_prob(z, eta)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    return None

if __name__ == "__main__":
    test_NormFlow()
    test_ConditionedNormFlow()
