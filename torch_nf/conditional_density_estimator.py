import torch
import torch.nn.functional as F
import numpy as np
import torch_nf.density_estimator as de
from torch_nf.error_formatters import format_type_err_msg, dbg_check
from collections import OrderedDict
import time


class ConditionalDensityEstimator(torch.nn.Module):
    def __init__(self, density_estimator, D_x, hidden_layers, dropout=False):
        super().__init__()
        self.density_estimator = density_estimator
        self.D_x = D_x
        self.D_params = density_estimator.D_params
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        layers = [
            ("linear1", torch.nn.Linear(D_x, self.hidden_layers[0])),
            ("tanh1", torch.nn.Tanh()),
        ]
        if self.dropout:
            layers.append(("dropout1", torch.nn.Dropout()))
        for i in range(1, len(self.hidden_layers)):
            layers.append(
                    ("linear%d" % (i + 1), 
                     torch.nn.Linear(
                         self.hidden_layers[i - 1], 
                         self.hidden_layers[i])))
            layers.append(("relu%d" % (i + 1), torch.nn.Tanh()))
            if self.dropout:
                layers.append(("dropout%d" % (i + 1), torch.nn.Dropout()))
        layers.append(
            ("linear%d" % (len(self.hidden_layers) + 1), 
             torch.nn.Linear(self.hidden_layers[-1], self.D_params))
        )

        layer_dict = OrderedDict(layers)
        self.param_net = torch.nn.Sequential(layer_dict)

    @property
    def density_estimator(self):
        return self.__density_estimator

    @density_estimator.setter
    def density_estimator(self, val):
        if type(val) not in [de.NormFlow, de.MoG]:
            raise TypeError(format_type_err_msg(self, "density_estimator", val, de.DensityEstimator))
        self.__density_estimator = val

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
        if type(self.density_estimator) == de.NormFlow:
            z, log_q_z = self.density_estimator(N=N, params=params, freeze_bn=freeze_bn)
        else:
            z, log_q_z = self.density_estimator(N=N, params=params)
        return z, log_q_z

    def log_prob(self, z, x):
        params = self.param_net(x)
        log_prob = self.density_estimator.log_prob(z, params)
        return log_prob
