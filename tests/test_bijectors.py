""" Test bijectors."""

import torch
import numpy as np
from torch_nf.bijectors import (
    Bijector,
    RealNVP,
    MAF,
    BatchNorm,
    ToSimplex,
    Affine,
    ToInterval,
)
from pytest import raises


def test_Bijector_init():
    D = 4
    bijector = Bijector(D)
    assert bijector.D == 4

    with raises(TypeError):
        Bijector("foo")

    with raises(ValueError):
        Bijector(-1)

    z = np.zeros((3, 4))
    params = np.zeros((5, 6))
    with raises(NotImplementedError):
        z, log_det = bijector(z, params)

    with raises(NotImplementedError):
        z, log_det = bijector.forward_and_log_det(z, params)

    with raises(NotImplementedError):
        z, log_det = bijector.inverse_and_log_det(z, params)
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
        real_nvp = RealNVP(D, "foo", 10)
    with raises(ValueError):
        real_nvp = RealNVP(D, -1, 10)

    with raises(TypeError):
        real_nvp = RealNVP(D, 2, "foo")

    with raises(TypeError):
        real_nvp = RealNVP(2, 2, 20, "foo")

    real_nvp = RealNVP(D, num_layers, num_units)
    D_theta = real_nvp.count_num_params()

    M = 10
    N = 5
    np.random.seed(0)
    torch.manual_seed(0)
    params = torch.tensor(np.random.normal(0.0, 0.1, (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = real_nvp(z_in, params)
    assert z.shape[0] == M and z.shape[1] == N and z.shape[2] == D
    assert log_det.shape[0] == M and log_det.shape[1] == N
    assert torch.eq(z[:, :, : D // 2], z_in[:, :, : D // 2]).all()
    assert not torch.eq(z[:, :, D // 2 :], z_in[:, :, D // 2 :]).all()

    z_inv, log_det_inv = real_nvp.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-2

    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    D_extra = 10
    params = torch.tensor(np.random.normal(0.0, 1.0, (M, D_theta + D_extra)))
    z, log_det = real_nvp(z_in, params)
    assert not torch.eq(z[:, :, : D // 2], z_in[:, :, : D // 2]).all()
    assert torch.eq(z[:, :, D // 2 :], z_in[:, :, D // 2 :]).all()

    # Odd D
    D = 5
    M = 20
    num_layers = 1
    num_units = 15
    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    D_theta = real_nvp.count_num_params()
    params = torch.tensor(np.random.normal(0.0, 0.1, (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = real_nvp(z_in, params)
    z_inv, log_det_inv = real_nvp.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-4
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-4

    # D = 8
    D = 8
    M = 20
    num_layers = 1
    num_units = 15
    real_nvp = RealNVP(D, num_layers, num_units, transform_upper=False)
    D_theta = real_nvp.count_num_params()
    params = torch.tensor(np.random.normal(0.0, 0.1, (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = real_nvp(z_in, params)
    z_inv, log_det_inv = real_nvp.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-4
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-4

    return None

def test_MAF():
    D = 4
    num_layers = 2
    num_units = 15
    maf = MAF(D, num_layers, num_units)
    assert maf.name == "MAF"
    assert maf.D == D
    assert maf.num_layers == num_layers
    assert maf.num_units == num_units

    maf = MAF(D, 6, 2000)
    assert maf.num_layers == 5
    assert maf.num_units == 1000

    maf = MAF(D, 3, 4)
    assert maf.num_units == 5

    with raises(TypeError):
        maf = MAF(D, "foo", 10)
    with raises(ValueError):
        maf = MAF(D, -1, 10)

    with raises(TypeError):
        maf = MAF(D, 2, "foo")

    num_layers = 3
    num_units=20
    maf = MAF(D, num_layers, num_units)
    for i, m in enumerate(maf.ms):
        if i==0 or i==(len(maf.ms)-1):
            assert(np.sum(m > D) == 0)
        else:
            assert(np.sum(m >= D) == 0)
        assert(np.sum(m < 1) == 0)
    D_ins = [D] + num_layers*[num_units]
    D_outs = num_layers*[num_units] + [D]
    for i, M in enumerate(maf.Ms):
        assert M.shape[0] == 1 and M.shape[1] == D_ins[i] and M.shape[2] == D_outs[i]

    # reverse fac
    maf = MAF(D, num_layers, num_units, fwd_fac=False)

    M = 50
    N = 20
    np.random.seed(0)
    torch.manual_seed(0)
    D_theta = maf.count_num_params()
    params = torch.tensor(np.random.normal(0.0, 1., (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = maf(z_in, params)
    assert z.shape[0] == M and z.shape[1] == N and z.shape[2] == D
    assert log_det.shape[0] == M and log_det.shape[1] == N
    assert not torch.eq(z, z_in).all()

    z_inv, log_det_inv = maf.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-6

    D = 20
    maf = MAF(D, 3, 100)
    M = 50
    N = 20
    np.random.seed(0)
    torch.manual_seed(0)
    D_theta = maf.count_num_params()
    params = torch.tensor(np.random.normal(0.0, .1, (M, D_theta)))
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = maf(z_in, params)
    assert z.shape[0] == M and z.shape[1] == N and z.shape[2] == D
    assert log_det.shape[0] == M and log_det.shape[1] == N
    assert not torch.eq(z, z_in).all()

    z_inv, log_det_inv = maf.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-6
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-6

    return None


def test_ToInterval():
    D = 4
    lb = float("-inf") * np.ones((D,))
    ub = float("inf") * np.ones((D,))
    interval = ToInterval(D, lb, ub)
    
    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D)))
    z, log_det = interval(z_in)
    z_inv, log_det_inv = interval.inverse_and_log_det(z)
    assert np.sum((z_in.numpy() - z.numpy()) ** 2) < 1e-10
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-10
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-10

    b = 0.5
    lb = -b*np.array([1., np.inf, 1, np.inf])
    ub = b*np.array([1., 1., np.inf, np.inf])
    interval = ToInterval(D, lb, ub)
    
    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(0.0, 2.0, (M, N, D)))
    z, log_det = interval(z_in)
    assert (z[:,:,0] > -1).all()
    assert (z[:,:,0] < 1).all()
    assert (z[:,:,1] < 1).all()
    assert (z[:,:,2] > -1).all()

    z_inv, log_det_inv = interval.inverse_and_log_det(z)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-4
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-4

    lb = -np.ones((D,))
    ub = np.ones((D+1,))
    with raises(ValueError):
        interval = ToInterval(D, lb, ub)

    lb = -np.ones((D,))
    ub = np.ones((D,))
    ub[3] = -2
    with raises(ValueError):
        interval = ToInterval(D, lb, ub)

    lb = '[-1,-1,-1,-1]'
    ub = np.ones((D,))
    with raises(TypeError):
        interval = ToInterval(D, lb, ub)
    lb = [-1,-1,-1,-1]
    interval = ToInterval(D, lb, ub)
    z, log_det = interval(z_in)
    z_inv, log_det_inv = interval.inverse_and_log_det(z)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-10

    lb = -np.ones((D,))
    ub = '[1,1,1,1]'
    with raises(TypeError):
        interval = ToInterval(D, lb, ub)
    ub = [1, 1, 1, 1]
    interval = ToInterval(D, lb, ub)
    z, log_det = interval(z_in)
    z_inv, log_det_inv = interval.inverse_and_log_det(z)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-10




    return None


def test_Affine():
    D = 4
    affine = Affine(D)
    assert affine.D == D

    M = 20
    N = 50
    D_theta = affine.count_num_params()
    params = torch.tensor(np.random.normal(0.0, 1.0, (M, D_theta))).float()
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D))).float()

    z, log_det = affine.forward_and_log_det(z_in, params)

    log_scale = params[:, :D]
    shift = params[:, D:]
    z_true = z_in * torch.exp(log_scale[:, None, :]) + shift[:, None, :]
    log_det_true = torch.sum(log_scale, dim=1, keepdim=True)
    assert np.sum((z.numpy() - z_true.numpy()) ** 2) < 1e-10
    assert np.sum((log_det.numpy() - log_det_true.numpy()) ** 2) < 1e-10

    z_inv, log_det_inv = affine.inverse_and_log_det(z, params)
    assert np.sum((z_in.numpy() - z_inv.numpy()) ** 2) < 1e-10
    assert np.sum((log_det.numpy() - log_det_inv.numpy()) ** 2) < 1e-10

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
    assert batch_norm.momentum == 1.0
    with raises(TypeError):
        batch_norm = BatchNorm(D, "foo")
    with raises(ValueError):
        batch_norm = BatchNorm(D, -1.0)

    with raises(TypeError):
        batch_norm = BatchNorm(D, 0.5, "foo")
    with raises(ValueError):
        batch_norm = BatchNorm(D, 0.5, -1.0)

    batch_norm = BatchNorm(D, 0.1, 1e-5)
    D_theta = batch_norm.count_num_params()
    assert D_theta == 0

    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(10.0, 1.0, (M, N, D))).float()
    z, log_det = batch_norm(z_in)
    last_mean = batch_norm.get_last_mean()
    last_alpha = batch_norm.get_last_alpha()
    z_mc = z_in - last_mean[None, None, :]
    z_true = (z_in - last_mean[None, None, :]) / last_alpha[None, None, :]
    assert np.sum((z.numpy() - z_true.numpy()) ** 2) < 1e-2
    assert np.isclose(log_det.numpy(), -np.sum(np.log(last_alpha.numpy())))

    z2, log_det = batch_norm(z_in, use_last=True)
    assert np.sum((z2.numpy() - z_true.numpy()) ** 2) < 1e-2

    z_inv, log_det_inv = batch_norm.inverse_and_log_det(z)
    assert np.isclose(log_det.numpy(), log_det_inv.numpy()).all()
    assert np.sum((z_inv.numpy() - z_in.numpy()) ** 2) < 1e-2

    return None


def test_ToSimplex():
    D = 4
    bij = ToSimplex(D)
    assert bij.name == "ToSimplex"
    assert bij.D == D

    M = 20
    N = 50
    z_in = torch.tensor(np.random.normal(0.0, 1.0, (M, N, D - 1))).float()
    z, log_det = bij(z_in)
    z_in = z_in.numpy()
    z = z.numpy()
    log_det = log_det.numpy()
    assert np.isclose(np.sum(z, 2), 1.0).all()
    expz = np.exp(z_in)
    sum_exp = np.sum(expz, 2)
    den = sum_exp + 1
    z_true = np.concatenate((expz, np.ones((M, N, 1))), axis=2) / np.expand_dims(den, 2)
    assert np.isclose(z_true, z).all()
    log_det_true = (
        np.log(1 - (sum_exp / (sum_exp + 1))) - D * np.log(sum_exp + 1) - np.sum(z, 2)
    )
    assert np.isclose(z_true, z).all()

    return None


if __name__ == "__main__":
    test_Bijector_init()
    test_RealNVP()
    test_MAF()
    test_Affine()
    test_ToInterval()
    test_BatchNorm()
    test_ToSimplex()
