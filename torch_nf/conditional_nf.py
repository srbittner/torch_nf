import torch
import torch.nn.functional as F
import numpy as np
from torch_nf.error_formatters import format_type_err_msg
from torch_nf.bijectors import RealNVP, BatchNorm
from collections import OrderedDict


class ConditionedNormFlow(torch.nn.Module):
    def __init__(self, nf, D_x, hidden_layers):
        super().__init__()
        self.nf = nf
        self.D_x = D_x
        self.hidden_layers = hidden_layers
        self.D_params = nf.D_params

        layers = [
            ("linear1", torch.nn.Linear(D_x, hidden_layers[0])),
            ("relu1", torch.nn.ReLU()),
        ]
        for i in range(1, len(hidden_layers)):
            layers.append(
                (
                    "linear%d" % (i + 1),
                    torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                )
            )
            layers.append(("relu%d" % (i + 1), torch.nn.ReLU()))
        layers.append(
            (
                "linear%d" % (len(hidden_layers) + 1),
                torch.nn.Linear(hidden_layers[-1], self.D_params),
            )
        )

        layer_dict = OrderedDict(layers)
        self.param_net = torch.nn.Sequential(layer_dict)

    def __call__(self, x, N=100):
        params = self.param_net(x)
        z, log_det = self.nf(params, N=N)
        return z, log_det


class NormFlow(object):
    def __init__(
        self,
        D,
        arch_type,
        num_stages=1,
        num_layers=2,
        num_units=None,
        support_layer=None,
    ):
        super().__init__()
        self._set_D(D)
        self._set_arch_type(arch_type)
        if self.arch_type is not "coupling":
            raise NotImplementedError()
        self._set_num_stages(num_stages)
        self._set_num_layers(num_layers)
        if num_units is None:
            num_units = max(2 * D, 15)
        self._set_num_units(num_units)
        self.D_params = self.count_num_params()

        self.bijectors = []
        for i in range(num_stages):
            self.bijectors.append(
                RealNVP(D, num_layers, num_units, transform_upper=True)
            )
            self.bijectors.append(BatchNorm(D))
            self.bijectors.append(
                RealNVP(D, num_layers, num_units, transform_upper=False)
            )
            if i < num_stages - 1:
                self.bijectors.append(BatchNorm(D))

        if support_layer is not None:
            self.bijectors.append(support_layer(D))

    def __call__(self, params, N=100):
        return self.forward(params, N)

    def forward(self, params, N=100):
        M = params.size(0)
        omega = np.random.normal(0.0, 1.0, (M, N, self.D))
        z = torch.tensor(omega).float()

        log_q_z = np.log(
            np.prod(np.exp((-np.square(omega)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2)
        )
        log_q_z = torch.tensor(log_q_z)

        for i, bijector in enumerate(self.bijectors):
            if bijector.name == "BatchNorm":
                z, log_det, params = bijector(z, params, use_last=(M == 1))
            else:
                z, log_det, params = bijector(z, params)
            log_q_z = log_q_z - log_det

        return z, log_q_z

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
        if self.arch_type == "coupling":
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
