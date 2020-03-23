""" Test exponential families."""

import numpy as np
import torch
from torch_nf.exponential_families import ExponentialFamily, MVN, Dirichlet
from torch_nf.bijectors import ToSimplex
from pytest import raises

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_exponential_family_init():
    """Test ExponentialFamily class initialization."""
    D = 4
    N = 100
    exp_fam = ExponentialFamily(D)
    assert exp_fam.D == D
    assert exp_fam.support_layer is None
    assert exp_fam.D_eta == D
    
    with raises(TypeError):
        exp_fam = ExponentialFamily('foo')
    with raises(ValueError):
        exp_fam = ExponentialFamily(0)
        
    with raises(TypeError):
        exp_fam = ExponentialFamily(4, int)

    with raises(NotImplementedError):
        exp_fam.sample_eta(N)

    mu = np.zeros((D,))
    with raises(NotImplementedError):
        exp_fam.mu_to_eta(mu)

    eta = np.zeros((D,))
    with raises(NotImplementedError):
        exp_fam.eta_to_mu(eta)

    z = np.zeros((D,))
    with raises(NotImplementedError):
        exp_fam.T(z)


    return None


def test_MVN():
    def eye_var(D, Sigma):
        eye_ref = np.expand_dims(np.eye(D), 0)
        return np.sum((Sigma - eye_ref)**2)

    Ds = [2, 5, 20]
    M = 20
    N = 50
    for D in Ds:
        ef = MVN(D)
        assert ef.D == D
        assert ef.D_eta == (D + (D * (D + 1) // 2))
        assert ef.support_layer is None

        eta = ef.sample_eta(N, sigma_mu=1., iw_df_fac=5)
        assert eta.shape[0] == N
        assert eta.shape[1] == ef.D_eta

        mu, Sigma = ef.eta_to_mu(eta)
        assert mu.shape[0] == N
        assert mu.shape[1] == D
        assert Sigma.shape[0] == N
        assert Sigma.shape[1] == D
        assert Sigma.shape[2] == D

        # Check forward and backward mappings for EF are inverses.
        _eta = ef.mu_to_eta(mu, Sigma)
        assert np.isclose(eta, _eta).all()
       
        # Check properties of p(eta).
        mu_var = np.var(mu)
        Sigma_eye_var = eye_var(D, Sigma)
        for i in range(10):
            eta2 = ef.sample_eta(N, sigma_mu=2., iw_df_fac=20)
            mu2, Sigma2 = ef.eta_to_mu(eta2)

            # Make sure sigma_mu controls variance of sampled mus.
            mu2_var = np.var(mu2)
            assert(mu_var < mu2_var)
            # Make sure iw_df_fac controls the adherance to I for Sigma.
            Sigma2_eye_var = eye_var(D, Sigma2)
            assert(Sigma_eye_var > Sigma2_eye_var)

        # Check output of T.
        z = torch.tensor(np.random.normal(0., 10., (M, N, D)))
        T_z = ef.T(z)

        T_z_true = np.zeros((M,N,ef.D_eta))
        T_z_true[:,:,:D] = z
        zzT = np.matmul(z[:, :, :, None], z[:, :, None, :])
        ind = D
        for i in range(D):
            for j in range(i,D):
                T_z_true[:,:,ind] = zzT[:,:,i,j]
                ind += 1
        assert(np.isclose(T_z.numpy(), T_z_true).all())

        # N=1 case.
        _ = ef.sample_eta(1, sigma_mu=1., iw_df_fac=5)

        # KL
        KLs = ef.KL(z, np.ones((M,N)), eta)
        assert(not (np.isnan(KLs).any()))
        
    return None


def test_Dirichlet():
    Ds = [2, 5, 20]
    M = 20
    N = 50
    for D in Ds:
        ef = Dirichlet(D)
        assert ef.D == D
        assert ef.D_eta == D + 1
        assert type(ef.support_layer(D)) == ToSimplex

        eta = ef.sample_eta(N)
        assert eta.shape[0] == N
        assert eta.shape[1] == ef.D_eta

        alpha = ef.eta_to_mu(eta)
        assert alpha.shape[0] == N
        assert alpha.shape[1] == D

        # Check forward and backward mappings for EF are inverses.
        _eta = ef.mu_to_eta(alpha)
        assert np.isclose(eta, _eta).all()

        # Check properties of p(eta).
        alpha_var = np.var(alpha)
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)
        for i in range(10):
            eta2 = ef.sample_eta(N, lb=0.1, ub=3.)
            alpha2 = ef.eta_to_mu(eta2)

            # Make sure bounds are modifying distribution.
            alpha2_var = np.var(alpha2)
            assert(alpha_var < alpha2_var)
            alpha2_min = np.min(alpha2)
            assert(alpha2_min < alpha_min)
            alpha2_max = np.max(alpha2)
            assert(alpha2_max > alpha_max)

        # Check output of T.
        z = torch.tensor(np.random.uniform(0.1, 3., (M, N, D)))
        T_z = ef.T(z)

        _T_z = np.log(z.numpy())
        log_h_z = np.sum(_T_z, axis=2)
        T_z_true = np.concatenate((_T_z, np.expand_dims(log_h_z, 2)), axis=2)
        assert(np.isclose(T_z.numpy(), T_z_true).all())

        # KL
        KLs = ef.KL(z, np.ones((M,N)), eta)
        assert(not (np.isnan(KLs).any()))
    return None


if __name__ == "__main__":
    test_exponential_family_init()
    test_MVN()
    test_Dirichlet()
