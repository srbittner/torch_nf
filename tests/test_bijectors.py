""" Test bijectors."""

import torch
import numpy as np
from torch_nf.bijectors import Bijector, RealNVP, BatchNorm, ToSimplex, Affine
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

    M = 10
    N = 5
    np.random.seed(0)
    torch.manual_seed(0)
    params = torch.tensor(np.random.normal(0., 0.1, (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0., 1., (M, N, D)))
    z, log_det = real_nvp(z_in, params)
    assert(z.shape[0] == M and z.shape[1] == N and z.shape[2] == D)
    assert(log_det.shape[0] == M and log_det.shape[1] == N)
    assert(torch.eq(z[:,:,:D//2], z_in[:,:,:D//2]).all())
    assert(not torch.eq(z[:,:,D//2:], z_in[:,:,D//2:]).all())

    z_inv, log_det_inv = real_nvp.inverse_and_log_det(z, params)
    assert(np.sum((z_in.numpy() - z_inv.numpy())**2) < 1e-2)

    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    D_extra = 10
    params = torch.tensor(np.random.normal(0., 1., (M, D_theta+D_extra)))
    z, log_det = real_nvp(z_in, params)
    assert(not torch.eq(z[:,:,:D//2], z_in[:,:,:D//2]).all())
    assert(torch.eq(z[:,:,D//2:], z_in[:,:,D//2:]).all())

    return None

def test_Affine():
    D = 4
    affine = Affine(D)
    assert affine.D == D

    M = 20
    N = 50
    D_theta = affine.count_num_params()
    params = torch.tensor(np.random.normal(0., 1., (M, D_theta))).float()
    z_in = torch.tensor(np.random.normal(0., 1., (M, N, D))).float()

    z, log_det = affine.forward_and_log_det(z_in, params)

    log_scale = params[:,:D]
    shift = params[:,D:]
    z_true = z_in * torch.exp(log_scale[:,None,:]) + shift[:,None,:]
    log_det_true = torch.sum(log_scale, dim=1, keepdim=True)
    assert(np.sum((z.numpy() - z_true.numpy())**2) < 1e-10)
    assert(np.sum((log_det.numpy() - log_det_true.numpy())**2) < 1e-10)
    
    z_inv, log_det_inv = affine.inverse_and_log_det(z, params)
    assert(np.sum((z_in.numpy() - z_inv.numpy())**2) < 1e-10)
    assert(np.sum((log_det.numpy() - log_det_inv.numpy())**2) < 1e-10)
    
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
    assert np.isclose(batch_norm.get_last_mean(), np.zeros(D)).all()
    assert np.isclose(batch_norm.get_last_alpha(), np.ones(D)).all()

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


    batch_norm = BatchNorm(D, .1, 1e-5)
    D_theta = batch_norm.count_num_params()
    assert(D_theta == 0)

    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(10., 1., (M, N, D))).float()
    z, log_det = batch_norm(z_in)
    last_mean = batch_norm.get_last_mean()
    last_alpha = batch_norm.get_last_alpha()
    z_mc = z_in - last_mean[None, None, :]
    z_true = (z_in - last_mean[None,None,:]) / last_alpha[None, None, :]
    assert(np.sum((z.numpy() - z_true.numpy())**2) <  1e-2)
    assert(np.isclose(log_det.numpy(), -np.sum(np.log(last_alpha.numpy()))))

    z2, log_det = batch_norm(z_in, use_last=True)
    assert(np.sum((z2.numpy() - z_true.numpy())**2) <  1e-2)

    z_inv, log_det_inv = batch_norm.inverse_and_log_det(z)
    assert(np.isclose(log_det.numpy(), log_det_inv.numpy()).all())
    assert(np.sum((z_inv.numpy() - z_in.numpy())**2) <  1e-2)

    return None

def test_ToSimplex():
    D = 4
    bij = ToSimplex(D)
    assert bij.name == "ToSimplex"
    assert bij.D == D

    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(0., 1., (M, N, D-1))).float()
    z, log_det = bij(z_in)
    z_in = z_in.numpy()
    z = z.numpy()
    log_det = log_det.numpy()
    assert(np.isclose(np.sum(z, 2), 1.).all())
    expz = np.exp(z_in)
    sum_exp = np.sum(expz, 2)
    den = sum_exp + 1
    z_true = np.concatenate((expz, np.ones((M, N, 1))), axis=2)/ np.expand_dims(den, 2)
    assert(np.isclose(z_true, z).all())
    log_det_true = np.log(1 - (sum_exp/(sum_exp+1))) - D*np.log(sum_exp+1) - np.sum(z, 2)
    assert(np.isclose(z_true, z).all())
    
    return None


if __name__ == "__main__":
    #test_Bijector_init()
    #test_RealNVP()
    test_Affine()
    #test_BatchNorm()
    #test_ToSimplex()
