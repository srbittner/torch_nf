""" Test conditional normalizing flows."""

import numpy as np
import torch
import torch_nf.bijectors as bij 
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

def get_MoG_params(params, K, D, lb=None,ub=None):
    beg_alpha = 0
    beg_mu = beg_alpha+K
    beg_Sigma_inv = beg_mu+(K*D)

    _alpha = params[beg_alpha:beg_mu]
    mu = np.reshape(params[beg_mu:beg_Sigma_inv], (K, D))
    _U = np.reshape(params[beg_Sigma_inv:], (K, D*(D+1)//2))

    # Softmax
    exp_alpha = np.exp(_alpha)
    alpha = exp_alpha / np.sum(exp_alpha)
    
    has_bounds = (lb is not None) and (ub is not None)
    if has_bounds:
        m = ((ub - lb)/2.)[None,:]
        c = ((ub + lb)/2.)[None,:]
        mu = m*np.tanh(mu) + c

    inds = np.triu_indices(D)
    U = np.zeros((K,D,D))
    for k in range(K):
        U[k,inds[0], inds[1]] = _U[k,:]
    U[:,range(D),range(D)] = np.exp(U[:,range(D),range(D)])
    if has_bounds:
        U[:,range(D),range(D)] = U[:,range(D),range(D)] / np.sqrt(m)
    Sigma_inv = np.matmul(np.transpose(U, (0,2,1)), U)

    return alpha, mu, Sigma_inv
    

def test_MoG():
    """Test NormFlow class initialization."""

    D = 4
    conditioner = False
    K = 1

    mog = de.MoG(D, conditioner, K)
    assert mog.D == D
    assert not conditioner
    assert mog.K == K

    for K in [1, 3]:
        conditioner = True
        M = 10
        lb = np.random.normal(-3., 0.01, (D,))
        ub = np.random.normal(3., 0.01, (D,))

        mog = de.MoG(D, conditioner, K, lb=lb, ub=ub)
        mog.count_num_params()

        params = torch.normal(0., 1., (M, mog.D_params))
        alpha, mu, Sigma_inv, Sigma_det = mog._get_MoG_params(params)

        # Test that alpha obeys softmax property.
        assert np.isclose(alpha.sum(1).numpy(), 1.).all()
        # Test that Sigma_mk is PSD.
        for m in range(M):
            for k in range(K):
                Sigma_inv_mk = Sigma_inv[m,k,:,:]
                assert (Sigma_inv_mk[range(D),range(D)] >= 0.).all()
                assert np.isclose(Sigma_inv_mk, Sigma_inv_mk.T).all()

        alpha_true, mu_true, Sigma_inv_true, Sigma_det_true = [], [], [], []
        for i in range(M):
            _alpha, _mu, _Sigma_inv = get_MoG_params(params[i,:].numpy(), K, D, lb=lb, ub=ub)
            alpha_true.append(_alpha)
            mu_true.append(_mu)
            Sigma_inv_true.append(_Sigma_inv)
            Sigma_det_true.append(1. / np.linalg.det(_Sigma_inv))
        alpha_true = np.array(alpha_true)
        mu_true = np.array(mu_true)
        Sigma_inv_true = np.array(Sigma_inv_true)
        Sigma_det_true = np.array(Sigma_det_true)
        
        print(alpha_true.shape, alpha.shape)
        assert np.isclose(alpha_true, alpha, rtol=1e-3).all()
        assert np.isclose(mu_true, mu, rtol=1e-3).all()
        assert np.isclose(Sigma_inv_true, Sigma_inv.numpy(), rtol=1e-3).all()
        assert np.isclose(Sigma_det_true, Sigma_det.numpy(), rtol=1e-3).all()

        z, log_q_z = mog.forward(params, N=10)

        log_p_z_np = mog.log_prob_np(z, params)
        assert(np.isclose(log_q_z, log_p_z_np, rtol=1e-4).all())

        log_p_z_torch = mog.log_prob(z, params)

        assert(np.isclose(log_p_z_torch.numpy(), log_p_z_np, rtol=1e-2).all())

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
        D, conditioner, arch_type, num_stages, num_layers, 10, bij.ToSimplex(D)
    )

    assert nf.num_units == 15
    assert issubclass(type(nf.support_layer), bij.Bijector)

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

    nf = de.NormFlow(D, True, 'coupling', 2, 2, 20, bij.ToSimplex(D))
    assert(issubclass(type(nf.bijectors[0]), bij.RealNVP))
    assert(issubclass(type(nf.bijectors[1]), bij.BatchNorm))
    assert(issubclass(type(nf.bijectors[2]), bij.RealNVP))
    assert(issubclass(type(nf.bijectors[3]), bij.BatchNorm))
    assert(issubclass(type(nf.bijectors[4]), bij.Affine))
    assert(issubclass(type(nf.bijectors[5]), bij.RealNVP))
    assert(issubclass(type(nf.bijectors[6]), bij.BatchNorm))
    assert(issubclass(type(nf.bijectors[7]), bij.RealNVP))
    assert(issubclass(type(nf.bijectors[8]), bij.BatchNorm))
    assert(issubclass(type(nf.bijectors[9]), bij.Affine))
    assert(issubclass(type(nf.bijectors[10]), bij.ToSimplex))

    nf = de.NormFlow(D, False, 'coupling', 2, 2, 20)
    assert(issubclass(type(nf.bijectors[0]), bij.RealNVP))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)
    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    nf = de.NormFlow(D, False, 'AR', num_layers=2, num_units=20)
    assert(issubclass(type(nf.bijectors[0]), bij.MAF))
    z, log_q_z = nf(N)
    log_q_z_inv = nf.log_prob(z)

    assert(np.sum(np.square(log_q_z.detach().numpy() - log_q_z_inv.detach().numpy())) < 1e-2)

    return None
    

if __name__ == "__main__":
    #test_DensityEstimator()
    #test_NormFlow()
    test_MoG()
