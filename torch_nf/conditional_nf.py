import torch
import torch.nn.functional as F
import numpy
from error_formatters import format_type_err_msg
from collections import OrderedDict

class ParameterNetwork(torch.nn.Module):
    def __init__(self, D_eta, hidden_layers, D_params):
        super(ParameterNetwork, self).__init__()
        self.D_eta = D_eta
        self.hidden_layers = hidden_layers
        self.D_params = D_params

        layers = [('linear1', torch.nn.Linear(D_eta, hidden_layers[0])), ('relu1', torch.nn.ReLU())]
        for i in range(1, len(hidden_layers)):
            layers.append(('linear%d' % (i+1), torch.nn.Linear(hidden_layers[i-1], hidden_layers[i])))
            layers.append(('relu%d' % (i+1), torch.nn.ReLU()))
        layers.append(('linear%d' % (len(hidden_layers)+1), torch.nn.Linear(hidden_layers[-1], D_params)))

        layer_dict = OrderedDict(layers)
        self.model = torch.nn.Sequential(layer_dict)

    def __call__(self, eta):
        return self.model(eta)

class NormFlow(object):
    def __init__(self, D, arch_type, num_stages=1, num_layers=2, num_units=None):
        super(NormFlow, self).__init__()
        self._set_D(D)
        self._set_arch_type(arch_type)
        self._set_num_stages(num_stages)
        self._set_num_layers(num_layers)
        if num_units is None:
            num_units = max(2 * D, 15)
        self._set_num_units(num_units)
        self.D_params = self.count_num_params()

    def __call__(self, params, N=100):
        return self.forward(params, N)

    def forward(self, params, N=100):
        omega = torch.empty(1, N, self.D).normal_(mean=0.0, std=1.0)
        z = omega
        param_ind = 0
        if self.arch_type == "coupling":

            lower, upper = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
            # upper | lower
            t_upper, s_upper, params = t_s_layer(
                lower, lower, params, self.D // 2, self.num_units
            )
            for i in range(self.num_layers - 1):
                t_upper, s_upper, params = t_s_layer(
                    t_upper, s_upper, params, self.num_units, self.num_units
                )
            t_upper, s_upper, params = t_s_layer(
                t_upper, s_upper, params, self.num_units, self.D // 2, relu=False
            )
            upper = t_upper + upper * torch.exp(s_upper)

            # lower | upper
            t_lower, s_lower, params = t_s_layer(
                upper, upper, params, self.D // 2, self.num_units
            )
            for i in range(self.num_layers - 1):
                t_lower, s_lower, params = t_s_layer(
                    t_lower, s_lower, params, self.num_units, self.num_units
                )
            t_lower, s_lower, params = t_s_layer(
                t_lower, s_lower, params, self.num_units, self.D // 2, relu=False
            )
            lower = t_lower + lower * torch.exp(s_lower)

            z = torch.cat([lower, upper], dim=1)
            log_det = torch.sum(s_upper, dim=1) + torch.sum(s_lower, dim=1)

            return z, log_det

        else:
            raise NotImplementedError()

    def _set_arch_type(self, arch_type):
        arch_types = ["coupling"]
        if type(arch_type) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", arch_type, str))
        if arch_type not in arch_types:
            raise ValueError('NormalizingFlow arch_type must be "coupling".')
        self.arch_type = arch_type

    def _set_D(self, D):
        if type(D) is not int:
            raise TypeError(format_type_err_msg(self, "D", D, int))
        elif D < 2:
            raise ValueError("NormalizingFlow D %d must be greater than 0." % D)
        self.D = D

    def _set_num_stages(self, num_stages):
        if type(num_stages) is not int:
            raise TypeError(format_type_err_msg(self, "num_stages", num_stages, int))
        elif num_stages < 1:
            raise ValueError(
                "NormalizingFlow num_stages %d must be greater than 0." % num_stages
            )
        self.num_stages = num_stages

    def _set_num_layers(self, num_layers):
        if type(num_layers) is not int:
            raise TypeError(format_type_err_msg(self, "num_layers", num_layers, int))
        elif num_layers < 1:
            raise ValueError(
                "NormalizingFlow num_layers arg %d must be greater than 0." % num_layers
            )
        self.num_layers = num_layers

    def _set_num_units(self, num_units):
        if type(num_units) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", num_units, int))
        elif num_units < 1:
            raise ValueError(
                "NormalizingFlow num_units %d must be greater than 0." % num_units
            )
        self.num_units = num_units

    def count_num_params(self,):
        return 2 * (
            2
            * self.num_stages
            * (
                self.D * self.num_units
                + self.D // 2
                + self.num_units
                + (self.num_layers - 1) * (self.num_units + 1) * self.num_units
            )
        )


def t_s_layer(x_t, x_s, params, D_in, D_out, relu=True):
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
