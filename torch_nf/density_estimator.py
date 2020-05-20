import torch
import torch.nn.functional as F
import numpy as np
from torch_nf.error_formatters import format_type_err_msg
from torch_nf.bijectors import RealNVP, MAF, BatchNorm, Affine, Bijector
from collections import OrderedDict
import time


class DensityEstimator(object):
    def __init__(
        self,
        D,
        conditioner=False,
    ):
        super().__init__()
        self.D = D
        self.conditioner = conditioner


    @property
    def D(self,):
        return self.__D

    @D.setter
    def D(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "D", val, int))
        elif val < 2:
            raise ValueError("DensityEstimator D %d must be greater than 1." % val)
        self.__D = val

    @property
    def conditioner(self,):
        return self.__conditioner

    @conditioner.setter
    def conditioner(self, val):
        if type(val) is not bool:
            raise TypeError(format_type_err_msg(self, "conditioner", val, bool))
        self.__conditioner = val


    def __call__(self, N=100, params=None):
        if not self.conditioner:
            return self.forward(self.params, N)
        else:
            return self.forward(params, N)

    def forward(self, params, N=100, freeze_bn=False):
        raise NotImplementedError()

    def log_prob(self, z, params=None):
        raise NotImplementedError()

    def count_num_params(self,):
        raise NotImplementedError()

    def _param_init(self,):
        raise NotImplementedError()

class MoG(DensityEstimator):
    def __init__(
        self,
        D,
        conditioner=False,
        K=1,
    ):
        super().__init__(D, conditioner)
        self.K = K
        
        """
        Bunch o' stuff.
        """

        self.count_num_params()

        if (not self.conditioner):
            self._param_init()

    @property
    def K(self,):
        return self.__K

    @K.setter
    def K(self, val):
        if type(val) is not int:
            raise TypeError(format_type_err_msg(self, "K", val, int))
        elif val < 1:
            raise ValueError("MoG K %d must be greater than 0." % val)
        self.__K = val

    def _param_init(self,):
        self.params = None
        return None
    
    def count_num_params(self,):
        return None



class NormFlow(DensityEstimator):
    def __init__(
        self,
        D,
        conditioner=False,
        arch_type="AR",
        num_stages=1,
        num_layers=2,
        num_units=15,
        support_layer=None,
    ):
        super().__init__(D, conditioner)
        self.arch_type = arch_type
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_units = num_units
        self.support_layer = support_layer

        self.bijectors = []

        if arch_type == "coupling":
            for i in range(num_stages):
                self.bijectors.append(
                    RealNVP(D, num_layers, num_units, transform_upper=True)
                )
                self.bijectors.append(BatchNorm(D))
                self.bijectors.append(
                    RealNVP(D, num_layers, num_units, transform_upper=False)
                )
                self.bijectors.append(BatchNorm(D))
                self.bijectors.append(Affine(D))
        elif arch_type == "AR":
            self.bijectors.append(MAF(D, self.num_layers, self.num_units, fwd_fac=True))
            self.bijectors.append(BatchNorm(D))
            self.bijectors.append(Affine(D))
        elif arch_type == "affine":
            self.bijectors.append(Affine(D))

        if support_layer is not None:
            if issubclass(type(support_layer), Bijector):
                self.bijectors.append(support_layer)
            else:
                raise TypeError("Support layer not Bijector.")

        self.count_num_params()

        if (not self.conditioner):
            self._param_init()

    @property
    def arch_type(self,):
        return self.__arch_type

    @arch_type.setter
    def arch_type(self, val):
        arch_types = ["coupling", "AR", "affine"]
        if type(val) is not str:
            raise TypeError(format_type_err_msg(self, "arch_type", val, str))
        if val not in arch_types:
            raise ValueError(
                'NormalizingFlow arch_type must be "coupling", "AR", or "affine".'
            )
        self.__arch_type = val

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

    def __call__(self, N=100, params=None, freeze_bn=False):
        if not self.conditioner:
            return self.forward(self.params, N, freeze_bn=freeze_bn)
        else:
            return self.forward(params, N, freeze_bn=freeze_bn)

    def _param_init(self,):
        self.params = torch.nn.init.xavier_normal_(
            torch.zeros(1, self.D_params, requires_grad=True)
        )
        return None

    def forward(self, params, N=100, freeze_bn=False):
        M = params.size(0)
        omega = np.random.normal(0.0, 1.0, (M, N, self.D))
        z = torch.tensor(omega).float()

        log_q_z = np.log(
            np.prod(np.exp((-np.square(omega)) / 2.0) / np.sqrt(2.0 * np.pi), axis=2)
        )
        log_q_z = torch.tensor(log_q_z)

        idx = 0  # parameter index
        for i, bijector in enumerate(self.bijectors):
            if bijector.name == "BatchNorm":
                z, log_det = bijector(z, use_last=freeze_bn)
            else:
                num_ps = (
                    bijector.count_num_params()
                )  # number of parameters for bijector
                if num_ps > 0:
                    z, log_det = bijector(z, params[:, idx : (idx + num_ps)])
                    idx += num_ps
                else:
                    z, log_det = bijector(z)
            log_q_z = log_q_z - log_det
        return z, log_q_z

    def inverse_and_log_det(self, z, params):
        num_bijectors = len(self.bijectors)
        z_size = z.size()
        idx = self.D_params
        sum_log_det = torch.zeros((z_size[0], z_size[1]))
        for i in range(num_bijectors - 1, -1, -1):
            bijector = self.bijectors[i]
            num_ps = bijector.count_num_params()
            if num_ps > 0:
                z, log_det = bijector.inverse_and_log_det(
                    z, params[:, (idx - num_ps) : idx]
                )
                idx -= num_ps
            else:
                z, log_det = bijector.inverse_and_log_det(z)
            sum_log_det += log_det
        return z, sum_log_det

    def log_prob(self, z, params=None):
        if not self.conditioner:
            z, sum_log_det = self.inverse_and_log_det(z, self.params)
        else:
            z, sum_log_det = self.inverse_and_log_det(z, params)
        log_q_z = torch.sum(-(z ** 2), axis=2) / 2.0 - self.D * np.log(
            np.sqrt(2.0 * np.pi)
        )
        return log_q_z - sum_log_det

    def count_num_params(self,):
        self.D_params = 0
        for bijector in self.bijectors:
            self.D_params += bijector.count_num_params()
    


