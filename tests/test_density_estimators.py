""" Test conditional normalizing flows."""

import numpy as np
import torch
#from torch_nf.conditional_nf import NormFlow, ConditionedNormFlow
import torch_nf.bijectors as tnfb #Bijector, RealNVP, MAF, BatchNorm, ToSimplex, Affine
import torch_nf.density_estimator as de

from pytest import raises

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def test_DensityEstimator():
    """Test DensityEstimator."""

    D = 4
    N = 100
    M = 10
    num_params = 8

    conditioner = False
    de1 = de.DensityEstimator(D, conditioner)
    assert de1.D == D
    assert not de1.conditioner

    conditioner = True
    de2 = de.DensityEstimator(D, conditioner)
    assert de2.D == D
    assert de2.conditioner

    z = np.random.normal((M, N, D))
    params = np.random.normal((M, num_params))
    with raises(NotImplementedError):
        de2(N, params)

    with raises(NotImplementedError):
        de2.log_prob(z, params)

    with raises(NotImplementedError):
        de2.count_num_params()

    return None

def test_MoG():
    """Test NormFlow class initialization."""

    D = 4
    conditioner = False
    K = 1
    mog = de.MoG(D, conditioner, K)
    assert mog.D == D
    assert not conditioner
    assert mog.K == K

   
    K = 3
    conditioner = True
    M = 50
    mog = de.MoG(D, conditioner, K)
    mog.count_num_params()

    params = torch.normal(0., 1., (M, mog.D_params))
    alpha, mu, Sigma = mog._get_MoG_params(params)

    assert np.isclose(alpha.sum(1).numpy(), 1.).all()
    for m in range(M):
        for k in range(K):
            Sigma_mk = Sigma[m,k,:,:]
            assert (Sigma_mk[range(D),range(D)] >= 0.).all()
            assert np.isclose(Sigma_mk, Sigma_mk.T).all()

    z = mog.forward(params, N=10)

    return None


def test_NormFlow():
    """Test NormFlow class initialization."""

    D = 4
    arch_type = "coupling"
    conditioner = False
    num_stages = 1
    num_layers = 2
    num_units = 30
    support_layer = None
    nf = de.NormFlow(
        D, conditioner, arch_type, num_stages, num_layers, num_units, support_layer
    )
    assert nf.arch_type == arch_type
    assert nf.num_stages == num_stages
    assert nf.num_layers == num_layers
    assert nf.num_units == num_units
    assert nf.support_layer == support_layer

    nf = de.NormFlow(
        D, conditioner, arch_type, num_stages, num_layers, 10, tnfb.ToSimplex(D)
    )

    assert nf.num_units == 15
    assert issubclass(type(nf.support_layer), tnfb.Bijector)

    with raises(TypeError):
        nf = de.NormFlow('foo', False, 'coupling', 1, 2, 20, None)
    with raises(ValueError):
        nf = de.NormFlow(-1, False, 'coupling', 1, 2, 20, None)

    with raises(TypeError):
        nf = de.NormFlow(4, False, 1, 1, 2, 20, None)
    with raises(ValueError):
        nf = de.NormFlow(4, False, 'foo', 1, 2, 20, None)

    with raises(TypeError):
        nf = de.NormFlow(4, 1, 'coupling', 1, 2, 20, None)

    with raises(TypeError):
        nf = de.NormFlow(4, False, 'coupling', 'foo', 2, 20, None)
    with raises(ValueError):
        nf = de.NormFlow(4, False, 'coupling', -1, 2, 20, None)

    with raises(TypeError):
        nf = de.NormFlow(4, False, 'coupling', 1, 'foo', 20, None)
    with raises(ValueError):
        nf = de.NormFlow(4, False, 'coupling', 1, -1, 20, None)

    with raises(TypeError):
        nf = de.NormFlow(4, False, 'coupling', 1, 2, 'foo', None)
    with raises(ValueError):
        nf = de.NormFlow(4, False, 'coupling', 1, 2, -1, None)

    with raises(TypeError):
        nf = de.NormFlow(4, False, 'coupling', 1, 2, 20, 'foo')

    nf = de.NormFlow(D, False, 'coupling', 1, 2, 20, None)

    N = 10
    z, log_q_z = nf(N)
    assert(z.shape[0] == 1 and z.shape[1] == N and z.shape[2] == D)
    assert(log_q_z.shape[0] == 1 and log_q_z.shape[1] == N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    nf = de.NormFlow(D, True, 'coupling', 2, 2, 20, tnfb.ToSimplex(D))
    assert(issubclass(type(nf.bijectors[0]), tnfb.RealNVP))
    assert(issubclass(type(nf.bijectors[1]), tnfb.BatchNorm))
    assert(issubclass(type(nf.bijectors[2]), tnfb.RealNVP))
    assert(issubclass(type(nf.bijectors[3]), tnfb.BatchNorm))
    assert(issubclass(type(nf.bijectors[4]), tnfb.Affine))
    assert(issubclass(type(nf.bijectors[5]), tnfb.RealNVP))
    assert(issubclass(type(nf.bijectors[6]), tnfb.BatchNorm))
    assert(issubclass(type(nf.bijectors[7]), tnfb.RealNVP))
    assert(issubclass(type(nf.bijectors[8]), tnfb.BatchNorm))
    assert(issubclass(type(nf.bijectors[9]), tnfb.Affine))
    assert(issubclass(type(nf.bijectors[10]), tnfb.ToSimplex))

    nf = de.NormFlow(D, False, 'coupling', 2, 2, 20)
    assert(issubclass(type(nf.bijectors[0]), tnfb.RealNVP))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    nf = de.NormFlow(D, False, 'AR', num_layers=2, num_units=20)
    assert(issubclass(type(nf.bijectors[0]), tnfb.MAF))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)

    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    return None
    

if __name__ == "__main__":
    test_DensityEstimator()
    test_NormFlow()
    test_MoG()
