""" Test exponential families."""

import numpy as np
from torch_nf.exponential_families import ExponentialFamily, MVN, Dirichlet
from torch_nf.bijectors import ToSimplex
from pytest import raises


def test_exponential_family_init():
    """Test ExponentialFamily class initialization."""
    D = 4
    N = 100
    exp_fam = ExponentialFamily(D)
    assert exp_fam.D == D
    assert exp_fam.support_layer is None
    assert exp_fam.D_eta == D

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
    Ds = [2, 5, 20]
    N = 50
    for D in Ds:
        ef = MVN(D)
        assert ef.D == D
        assert ef.D_eta == (D + (D * (D + 1) // 2))
        assert ef.support_layer is None

        eta = ef.sample_eta(N)
        assert eta.shape[0] == N
        assert eta.shape[1] == ef.D_eta

        mu, Sigma = ef.eta_to_mu(eta)
        assert mu.shape[0] == N
        assert mu.shape[1] == D
        assert Sigma.shape[0] == N
        assert Sigma.shape[1] == D
        assert Sigma.shape[2] == D

        _eta = ef.mu_to_eta(mu, Sigma)
        assert np.isclose(eta, _eta).all()
    return None


def test_Dirichlet():
    Ds = [2, 5, 20]
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

        _eta = ef.mu_to_eta(alpha)
        assert np.isclose(eta, _eta).all()

    return None


if __name__ == "__main__":
    test_exponential_family_init()
    test_MVN()
    test_Dirichlet()
