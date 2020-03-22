import torch
import torch.nn.functional as F
import numpy as np
from torch_nf.error_formatters import format_type_err_msg

class Bijector(object):
    def __init__(self,):
        super().__init__()

    def __call__(self, z, params):
        return self.forward_and_log_det(z, params)

    def forward_and_log_det(self, z, params):
        raise NotImplementedError()

class RealNVP(Bijector):
    def __init__(self, D, num_layers, num_units, transform_upper=True):
        super().__init__()
        self.name = "RealNVP"
        self.D = D
        self.num_layers = num_layers
        self.num_units = num_units
        self.transform_upper = transform_upper

    def forward_and_log_det(self, z, params):
        if self.transform_upper:
            z1, z2 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        else:
            z2, z1 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        # upper | lower
        t, s, params = self._t_s_layer(
            z1, z1, params, self.D // 2, self.num_units
        )
        for i in range(self.num_layers - 1):
            t, s, params = self._t_s_layer(
                t, s, params, self.num_units, self.num_units
            )
        t, s, params = self._t_s_layer(
            t, s, params, self.num_units, self.D // 2, relu=False
        )
        z2 = t + z2 * torch.exp(s)

        if self.transform_upper:
            z = torch.cat([z1, z2], dim=2)
        else:
            z = torch.cat([z2, z1], dim=2)

        log_det = torch.sum(s, dim=2)
        return z, log_det, params

    def _t_s_layer(self, x_t, x_s, params, D_in, D_out, relu=True):
        param_ind = 0
        _num_param_weight = D_in * D_out
        t_weight = params[:, param_ind : (param_ind + _num_param_weight)].view(
            -1, D_in, D_out
        )
        param_ind += _num_param_weight
        s_weight = params[:, param_ind : (param_ind + _num_param_weight)].view(
            -1, D_in, D_out
        )
        param_ind += _num_param_weight

        _num_param_bias = D_out
        t_bias = params[:, param_ind : (param_ind + _num_param_bias)].view(
            -1, 1, _num_param_bias
        )
        param_ind += _num_param_bias
        s_bias = params[:, param_ind : (param_ind + _num_param_bias)].view(
            -1, 1, _num_param_bias
        )
        param_ind += _num_param_bias

        t = torch.matmul(x_t, t_weight) + t_bias
        s = torch.matmul(x_s, s_weight) + s_bias
        if relu:
            t = F.relu(t)
            s = F.relu(s)
        return t, s, params[:, param_ind:]

class BatchNorm(Bijector):
    def __init__(self, D, momentum=0.1):
        super().__init__()
        self.name = "BatchNorm"
        self.D = D
        self.momentum = momentum
        self.eps = 1e-5
        self.batch_norm = torch.nn.BatchNorm1d(D, eps=self.eps, momentum=momentum, affine=False)
        self.last_mean = None
        self.last_alpha = None

    def __call__(self, z, params, use_last=False):
        return self.forward_and_log_det(z, params, use_last=use_last)

    def forward_and_log_det(self, z, params, use_last=False):
        if (use_last):
            alpha = self.last_alpha
            z = (z - self.last_mean) / alpha

        else:
            z_size = z.size()
            z_mean = torch.mean(z, dim=[0,1], keepdim=True)
            z_mc = z - z_mean

            z_norm = self.batch_norm(z.view(-1, self.D))
            z_norm = z_norm.view(z_size[0], z_size[1], self.D)

            alpha = torch.mean(z_mc / z_norm, dim=[0,1], keepdim=True)

            self.last_mean = z_mean[0,0]
            self.last_alpha = alpha
        
        log_det = -torch.sum(torch.log(alpha))
        return z, log_det, params


class ToSimplex(Bijector):
    def __init__(self, D):
        super().__init__()
        self.name = "ToSimplex"
        self.D = D

    def __call__(self, z, params, use_last=False):
        return self.forward_and_log_det(z, params, use_last=use_last)

    def forward_and_log_det(self, z, params, use_last=False):

        ex = torch.exp(z)
        sum_ex = torch.sum(ex, dim=2)
        den = sum_ex + 1.0
        log_det = (
                torch.log(1.0 - (sum_ex / den))
                - self.D * torch.log(den)
                + torch.sum(z, axis=2)
        )
        z = torch.cat((ex / den[:,:,None], 1.0 / den[:,:,None]), axis=2)

        return z, log_det, params


