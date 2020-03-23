""" Test bijectors."""

import torch
import numpy as np
from torch_nf.bijectors import Bijector, RealNVP, BatchNorm, ToSimplex
from pytest import raises


def test_Bijector_init():
    D = 4
    bijector = Bijector(D)
    assert(bijector.D == 4)

    with raises(TypeError):
        Bijector('foo')

    with raises(ValueError):
        Bijector(-1)

    z = np.zeros((3, 4))
    params = np.zeros((5, 6))
    with raises(NotImplementedError):
        z, log_det = bijector(z, params)

    with raises(NotImplementedError):
        z, log_det = bijector.forward_and_log_det(z, params)

    return None


def test_RealNVP():
    D = 4
    num_layers = 2
    num_units = 15
    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    assert real_nvp.name == "RealNVP"
    assert real_nvp.D == D
    assert real_nvp.num_layers == num_layers
    assert real_nvp.num_units == num_units
    assert not real_nvp.transform_upper

    real_nvp = RealNVP(D, 6, 2000)
    assert real_nvp.num_layers == 5
    assert real_nvp.num_units == 1000

    real_nvp = RealNVP(D, 3, 10)
    assert real_nvp.num_units == 15

    with raises(TypeError):
        real_nvp = RealNVP(D, 'foo', 10)
    with raises(ValueError):
        real_nvp = RealNVP(D, -1, 10)

    with raises(TypeError):
        real_nvp = RealNVP(D, 2, 'foo')

    with raises(TypeError):
        real_nvp = RealNVP(2, 2, 20, 'foo')

    real_nvp = RealNVP(D, num_layers, num_units)
    D_theta = real_nvp.count_num_params()

    M = 20
    N = 50
    params = torch.tensor(np.random.normal(0., 1., (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0., 1., (M, N, D)))
    z, log_det, params = real_nvp(z_in, params)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_det.shape[0] == M and log_det.shape[1] == N)
    assert(params.shape[0] == M and params.shape[1] == 0)
    assert(torch.eq(z[:,:,:D//2], z_in[:,:,:D//2]).all())
    assert(not torch.eq(z[:,:,D//2:], z_in[:,:,D//2:]).all())

    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    D_extra = 10
    params = torch.tensor(np.random.normal(0., 1., (M, D_theta+D_extra)))
    z, log_det, params = real_nvp(z_in, params)
    assert(params.shape[0] == M and params.shape[1] == D_extra)
    assert(not torch.eq(z[:,:,:D//2], z_in[:,:,:D//2]).all())
    assert(torch.eq(z[:,:,D//2:], z_in[:,:,D//2:]).all())

    return None


def test_BatchNorm():
    D = 4
    momentum = 0.05
    eps = 1e-7
    batch_norm = BatchNorm(D, momentum, eps)
    assert batch_norm.name == "BatchNorm"
    assert batch_norm.D == D
    assert batch_norm.momentum == momentum
    assert batch_norm.eps == eps
    assert batch_norm.get_last_mean() == None
    assert batch_norm.get_last_alpha() == None

    batch_norm = BatchNorm(D, 1.01)
    assert(batch_norm.momentum == 1.)
    with raises(TypeError):
        batch_norm = BatchNorm(D, 'foo')
    with raises(ValueError):
        batch_norm = BatchNorm(D, -1.)

    with raises(TypeError):
        batch_norm = BatchNorm(D, 0.5, 'foo')
    with raises(ValueError):
        batch_norm = BatchNorm(D, 0.5, -1.)



def test_ToSimplex():
    D = 4
    bij = ToSimplex(D)
    assert bij.name == "ToSimplex"
    assert bij.D == D


if __name__ == "__main__":
    #test_Bijector_init()
    test_RealNVP()
    #test_BatchNorm()
    #test_ToSimplex()
