import torch
import torch.nn.functional as F
import numpy as np
import torch_nf.density_estimator as de
from torch_nf.error_formatters import format_type_err_msg
from torch_nf.bijectors import RealNVP, MAF, BatchNorm, Affine, Bijector
from collections import OrderedDict
import time


class ConditionedNormFlow(torch.nn.Module):
    def __init__(self, nf, D_x, hidden_layers, dropout=False):
        super().__init__()
        self.nf = nf
        self.D_x = D_x
        self.D_params = nf.D_params
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        hl = hidden_layers

        layers = [
            ("linear1", torch.nn.Linear(D_x, hidden_layers[0])),
            ("tanh1", torch.nn.Tanh()),
        ]
        if self.dropout:
            layers.append(("dropout1", torch.nn.Dropout()))
        for i in range(1, len(hl)):
            layers.append(("linear%d" % (i + 1), torch.nn.Linear(hl[i - 1], hl[i])))
            layers.append(("relu%d" % (i + 1), torch.nn.Tanh()))
            if self.dropout:
                layers.append(("dropout%d" % (i + 1), torch.nn.Dropout()))
        layers.append(
            ("linear%d" % (len(hl) + 1), torch.nn.Linear(hl[-1], self.D_params))
        )

        layer_dict = OrderedDict(layers)
        self.param_net = torch.nn.Sequential(layer_dict)

    @property
    def nf(self):
        return self.__nf

    @nf.setter
    def nf(self, val):
        if type(val) is not de.NormFlow:
            raise TypeError(format_type_err_msg(self, "nf", val, de.NormFlow))
        self.__nf = val

    @property
    def D_x(self,):
        return self.__D_x

    @D_x.setter
    def D_x(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "D_x", val, int))
        elif val < 1:
            raise ValueError("D_x %d must be greater than 0." % val)
        self.__D_x = val

    @property
    def D_params(self,):
        return self.__D_params

    @D_params.setter
    def D_params(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "D_params", val, int))
        elif val < 1:
            raise ValueError("D_params %d must be greater than 0." % val)
        self.__D_params = val

    @property
    def hidden_layers(self,):
        return self.__hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, val):
        if type(val) is not list:
            raise TypeError(format_type_err_msg(self, "hidden_layers", val, list))
        for i, num_units in enumerate(val):
            if type(num_units) is not int:
                raise TypeError(
                    format_type_err_msg(self, "hidden_layers[%d]" % i, val, int)
                )
            if num_units < 1:
                raise ValueError("Hidden unit counts must be positive.")
        self.__hidden_layers = val

    def __call__(self, x, N=100, freeze_bn=False):
        params = self.param_net(x)
        z, log_q_z = self.nf(N=N, params=params, freeze_bn=freeze_bn)
        return z, log_q_z

    def log_prob(self, z, x):
        params = self.param_net(x)
        log_prob = self.nf.log_prob(z, params)
        return log_prob

def dbg_check(tensor, name):
    num_elems = 1
    for dim in tensor.shape:
        num_elems *= dim
    num_infs = torch.sum(torch.isinf(tensor)).item()
    num_nans = torch.sum(torch.isnan(tensor)).item()

    print(
        name, "infs %d/%d" % (num_infs, num_elems), "nans %d/%d" % (num_nans, num_elems)
    )
    return None
