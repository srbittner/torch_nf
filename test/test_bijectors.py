""" Test bijectors."""

import numpy as np
from torch_nf.bijectors import Bijector, RealNVP, BatchNorm, ToSimplex
from pytest import raises

def test_Bijector_init():
    bijector = Bijector()
    z = np.zeros((3, 4))
    params = np.zeros((5, 6))
    with raises(NotImplementedError):
        z, log_det = bijector(z, params)

    with raises(NotImplementedError):
        z, log_det = bijector.forward_and_log_det(z, params)
        
    return  None

def test_RealNVP():
    D = 4
    num_layers = 2
    num_units = 15
    real_nvp = RealNVP(D, num_layers, num_units)
    assert(real_nvp.name == "RealNVP")
    assert(real_nvp.D == D)
    assert(real_nvp.num_layers == num_layers)
    assert(real_nvp.num_units == num_units)
    return None

def test_BatchNorm():
    D = 4
    momentum = 0.05
    batch_norm = BatchNorm(D, momentum)
    assert(batch_norm.name == "BatchNorm")
    assert(batch_norm.D == D)
    assert(batch_norm.momentum == momentum)
    assert(batch_norm.eps == 1e-5)
    assert(batch_norm.last_mean == None)
    assert(batch_norm.last_alpha == None)

    return None

def test_ToSimplex():
    D = 4
    bij = ToSimplex(D)
    assert(bij.name == "ToSimplex")
    assert(bij.D == D)

if __name__ == "__main__":
    test_bijector_init()
    test_RealNVP()
    test_BatchNorm()
    test_ToSimplex()
