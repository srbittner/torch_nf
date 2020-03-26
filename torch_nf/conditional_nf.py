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
        self.D_params = nf.D_params
        self.hidden_layers = hidden_layers

        hl = hidden_layers

        layers = [
            ("linear1", torch.nn.Linear(D_x, hidden_layers[0])),
            ("relu1", torch.nn.ReLU()),
        ]
        for i in range(1, len(hl)):
            layers.append(("linear%d" % (i + 1), torch.nn.Linear(hl[i - 1], hl[i]),))
            layers.append(("relu%d" % (i + 1), torch.nn.ReLU()))
        layers.append(
            ("linear%d" % (len(hl) + 1), torch.nn.Linear(hl[-1], self.D_params),)
        )

        layer_dict = OrderedDict(layers)
        self.param_net = torch.nn.Sequential(layer_dict)

    @property
    def nf(self):
        return self.__nf

    @nf.setter
    def nf(self, val):
        if type(val) is not NormFlow:
            raise TypeError(format_type_err_msg(self, "nf", val, NormFlow))
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

    def __call__(self, x, N=100):
        params = self.param_net(x)
        z, log_det = self.nf(N=N, params=params)
        return z, log_det


class NormFlow(object):
    def __init__(
        self,
        D,
        arch_type,
        conditioner=False,
        num_stages=1,
        num_layers=2,
        num_units=None,
        support_layer=None,
    ):
        super().__init__()
        self.D = D
        self.arch_type = arch_type
        self.conditioner = conditioner
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_units = num_units
        self.support_layer = support_layer

        self.bijectors = []
        for i in range(num_stages):
            self.bijectors.append(
                RealNVP(D, num_layers, num_units, transform_upper=True)
            )
            self.bijectors.append(BatchNorm(D))
            self.bijectors.append(
                RealNVP(D, num_layers, num_units, transform_upper=False)
            )
            self.bijectors.append(BatchNorm(D))
            # if i < num_stages - 1:
            #    self.bijectors.append(BatchNorm(D))

        if support_layer is not None:
            self.bijectors.append(support_layer(D))

        self.count_num_params()

        if not self.conditioner:
            self.params = torch.nn.init.xavier_normal_(
                torch.zeros(1, self.D_params, requires_grad=True)
            )

    @property
    def D(self,):
        return self.__D

    @D.setter
    def D(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "D", val, int))
        elif val < 2:
            raise ValueError("NormalizingFlow D %d must be greater than 0." % val)
        self.__D = val

    @property
    def arch_type(self,):
        return self.__arch_type

    @arch_type.setter
    def arch_type(self, val):
        arch_types = ["coupling"]
        if type(val) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", val, str))
        if val not in arch_types:
            raise ValueError('NormalizingFlow arch_type must be "coupling".')
        self.__arch_type = val

    @property
    def conditioner(self,):
        return self.__conditioner

    @conditioner.setter
    def conditioner(self, val):
        if type(val) is not bool:
            raise TypeError(format_type_err_msg(self, "conditioner", val, bool))
        self.__conditioner = val

    @property
    def num_stages(self,):
        return self.__num_stages

    @num_stages.setter
    def num_stages(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "num_stages", val, int))
        elif val < 1:
            raise ValueError(
                "NormalizingFlow num_stages %d must be greater than 0." % val
            )
        self.__num_stages = val

    @property
    def num_layers(self,):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "num_layers", val, int))
        elif val < 1:
            raise ValueError(
                "NormalizingFlow num_layers arg %d must be greater than 0." % val
            )
        self.__num_layers = val

    @property
    def num_units(self,):
        return self.__num_units

    @num_units.setter
    def num_units(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "num_units", val, int))
        elif val < 1:
            raise ValueError(
                "NormalizingFlow num_units %d must be greater than 0." % val
            )
        elif val < 15:
            print(
                "Warning: NormFlow.num_layers set to minimum of 15 (received %d)." % val
            )
            self.__num_units = 15
        else:
            self.__num_units = val

    def __call__(self, N=100, params=None):
        if not self.conditioner:
            return self.forward(self.params, N)
        else:
            return self.forward(params, N)

    def forward(self, params, N=100):
        M = params.size(0)
        omega = np.random.normal(0.0, 1.0, (M, N, self.D))
        z = torch.tensor(omega).float()

        log_q_z = np.log(
            np.prod(np.exp((-np.square(omega)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2)
        )
        log_q_z = torch.tensor(log_q_z)

        idx = 0 # parameter index
        for i, bijector in enumerate(self.bijectors):
            if bijector.name == "BatchNorm":
                z, log_det = bijector(z, use_last=(M == 1))
            else:
                num_ps = bijector.count_num_params() # number of parameters for bijector
                if num_ps > 0:
                    z, log_det = bijector(z, params[:, idx:(idx+num_ps)])
                    idx += num_ps
                else:
                    z, log_det = bijector(z)
            log_q_z = log_q_z - log_det
        return z, log_q_z

    def inverse(self, z, params):
        num_bijectors = len(self.bijectors)
        idx = self.count_num_params()
        sum_log_det = 0.0
        for i in range(num_bijectors + 1, -1, -1):
            bijector = self.bijectors[i]
            num_ps = bijector.count_num_params()
            if num_ps > 0:
                z, log_det = bijector.inverse(z, params[:, (idx-num_ps):idx])
                idx -= num_ps
            else:
                z, log_det = bijector.inverse(z)
            sum_log_det += log_det
        return z, sum_log_det

    def log_prob(self, z, params):
        z, sum_log_det = self.inverse(z, params)
        log_q_z = torch.log(
            torch.prod(
                torch.exp((-torch.square(z)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2
            )
        )
        return log_q_z

    def count_num_params(self,):
        self.D_params = 0
        for bijector in self.bijectors:
            self.D_params += bijector.count_num_params()


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
