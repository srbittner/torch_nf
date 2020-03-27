import torch
import torch.nn.functional as F
import numpy as np
from torch_nf.error_formatters import format_type_err_msg


class Bijector(object):
    """Base class for bijectors to be composed into normalizing flows. 

    :param D: Dimensionality of the bijection.
    :type D: int
    """

    def __init__(self, D):
        super().__init__()
        self.D = D

    @property
    def D(self,):
        return self.__D

    @D.setter
    def D(self, val):
        if type(val) is not int:
            raise (TypeError(format_type_err_msg(self, "D", val, int)))
        elif val < 1:
            raise ValueError("Bijector dimensionality must be positive.")
        self.__D = val

    def __call__(self, z, params):
        return self.forward_and_log_det(z, params)

    def forward_and_log_det(self, z, params):
        """Run the bijector forward on the input and compute the log det jac.

        The input z should have two batch dimensions.  The first, of size M,
        should match the leading dimension of parameters params.  The second
        batch dimension should be the number of samples per parameterization N.
        The earliest elements of the second dimension of z shall be used to 
        parameterize this bijection.

        :param z: Input to the bijector (M, N, D).
        :type z: torch.tensor
        :param params: Parameterization of the bijector (M, >|theta|).
        :type params: torch.tensor
        """
        raise NotImplementedError()

    def inverse_and_log_det(self, z, params):
        """Run the bijector backwards and compute the log det jac.

        The input z should have two batch dimensions.  The first, of size M,
        should match the leading dimension of parameters params.  The second
        batch dimension should be the number of samples per parameterization N.
        The earliest elements of the second dimension of z shall be used to 
        parameterize this bijection. 

        :param z: Input to the bijector (M, N, D).
        :type z: torch.tensor
        :param params: Parameterization of the bijector (M, >|theta|).
        :type params: torch.tensor
        """
        raise NotImplementedError()

    def count_num_params(self,):
        """Return the number of parameters for the bijector.

        :return: Number of parameters of the bijector.
        :rtype: int
        """
        return 0

class RealNVP(Bijector):
    """RealNVP bijector.

    A fully connected neural network parameterizes the conditional affine
    transformation of half of the dimensions of the input z2 on the other half z1.
    The upper half is conditioned on the lower half if transform_upper=True,
    and the converse otherwise.

    :param D: Dimensionality of the bijection.
    :type D: int
    :param num_layers: Number of layers in the neural network for p(z2 | z1).
    :type num_layers: int
    :param num_units: Number of hidden units per layer in network for p(z2 | z1).
    :type num_units: int
    :param transform_upper: z2 is up-half, and z1 is low-half, default True.
    :type transform_upper: bool, optional

    """

    def __init__(self, D, num_layers, num_units, transform_upper=True):
        super().__init__(D)
        self.name = "RealNVP"
        self.num_layers = num_layers
        self.num_units = num_units
        self.transform_upper = transform_upper

    @property
    def num_layers(self,):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, val):
        if type(val) is not int:
            raise (TypeError(format_type_err_msg(self, "num_layers", val, int)))
        elif val < 1:
            raise ValueError("RealNVP.num_layers must be positive.")
        elif val > 5:
            print(
                "Warning: RealNVP.num_layers set to maximum of 5 (received %d)." % val
            )
            self.__num_layers = 5
        else:
            self.__num_layers = val

    @property
    def num_units(self,):
        return self.__num_units

    @num_units.setter
    def num_units(self, val):
        if type(val) is not int:
            raise (TypeError(format_type_err_msg(self, "num_units", val, int)))
        elif val < 15:
            print("Warning: num_units set to minimum of 15 (received %d)." % val)
            self.__num_units = 15
        elif val > 1000:
            print("Warning: num_units set to maximum of 1,000 (received %d)." % val)
            self.__num_units = 1000
        else:
            self.__num_units = val

    @property
    def transform_upper(self,):
        return self.__transform_upper

    @transform_upper.setter
    def transform_upper(self, val):
        if type(val) is not bool:
            raise (TypeError(format_type_err_msg(self, "transform_upper", val, bool)))
        self.__transform_upper = val

    def forward_and_log_det(self, z, params):
        """Forward transform of the RealNVP and log determinant of the jacobian.

        The shift and scale parameters of the upper (lower) half are outputs 
        of two separate neural networks which take the lower (upper) half
        as input.

        :param z: Input to the bijector (M, N, D).
        :type z: torch.tensor
        :param params: Parameterization of the bijector (M, >|theta|).
        :type params: torch.tensor
        """
        if self.transform_upper:
            z1, z2 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        else:
            z2, z1 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        # upper | lower
        t, s, params = self._t_s_layer(z1, z1, params, self.D // 2, self.num_units)
        for i in range(self.num_layers - 1):
            t, s, params = self._t_s_layer(t, s, params, self.num_units, self.num_units)
        t, s, params = self._t_s_layer(
            t, s, params, self.num_units, self.D // 2, relu=False
        )
        z2 = t + z2 * torch.exp(s)

        if self.transform_upper:
            z = torch.cat([z1, z2], dim=2)
        else:
            z = torch.cat([z2, z1], dim=2)

        log_det = torch.sum(s, dim=2)
        return z, log_det

    def inverse_and_log_det(self, z, params):
        if self.transform_upper:
            z1, z2 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        else:
            z2, z1 = z[:, :, : self.D // 2], z[:, :, self.D // 2 :]
        # upper | lower
        t, s, params = self._t_s_layer(z1, z1, params, self.D // 2, self.num_units)
        for i in range(self.num_layers - 1):
            t, s, params = self._t_s_layer(t, s, params, self.num_units, self.num_units)
        t, s, params = self._t_s_layer(
            t, s, params, self.num_units, self.D // 2, relu=False
        )
        z2 = (z2 - t) / torch.exp(s)

        if self.transform_upper:
            z = torch.cat([z1, z2], dim=2)
        else:
            z = torch.cat([z2, z1], dim=2)

        log_det = torch.sum(s, dim=2)
        return z, log_det



    def _t_s_layer(self, x_t, x_s, params, D_in, D_out, relu=True):
        """One layer of the neural network for shift and scale operations t and s. 

        :param x_t: Input to layer of shift neural network (M, N, D_in).
        :type x_t: torch.tensor
        :param x_s: Input to layer of shift neural network (M, N, D_in).
        :type x_s: torch.tensor
        :param params: Parameterization of the bijector (M, >|theta|).
        :type params: torch.tensor
        :param D_in: Dimensionality of input to networks.
        :type D_in: int
        :param D_out: Dimensionality of output of networks.
        :type D_out: int
        :param relu: Pass linear network operation through relu nonlinearity, default True.
        :type D_out: bool, optional
        """
        idx = 0
        num_ps = D_in * D_out
        t_weight = params[:, idx : (idx + num_ps)].view(
            -1, D_in, D_out
        )
        idx += num_ps
        s_weight = params[:, idx : (idx + num_ps)].view(
            -1, D_in, D_out
        )
        idx += num_ps

        num_ps = D_out
        t_bias = params[:, idx : (idx + num_ps)].view(
            -1, 1, num_ps
        )
        idx += num_ps
        s_bias = params[:, idx : (idx + num_ps)].view(
            -1, 1, num_ps
        )
        idx += num_ps

        t = torch.matmul(x_t, t_weight) + t_bias
        s = torch.matmul(x_s, s_weight) + s_bias
        if relu:
            t = F.relu(t)
            s = F.relu(s)
        return t, s, params[:, idx:]

    def count_num_params(self,):
        """Return the number of parameters for the bijector.

        :return: Number of parameters of the bijector.
        :rtype: int
        """
        return 2 * (
            self.D * self.num_units
            + self.D // 2
            + self.num_units
            + (self.num_layers - 1) * (self.num_units + 1) * self.num_units
        )

class Affine(Bijector):
    """Affine bijector.

    A scale and shift of each dimension.
    :param D: Dimensionality of the bijection.
    :type D: int
    """
    def __init__(self, D):
        super().__init__(D)
        self.name = "Affine"

    def forward_and_log_det(self, z, params):
        idx = 0

        num_ps = self.D
        scale = torch.exp(params[:,idx:(idx+num_ps)])
        idx += num_ps

        num_ps = self.D
        shift = params[:,idx:(idx+num_ps)]
        idx += num_ps

        scale = scale[:,None,:] 
        shift = shift[:,None,:] 

        z = scale*z + shift
        log_det = torch.sum(torch.log(scale), axis=2)

        return z, log_det

    def inverse_and_log_det(self, z, params):
        idx = 0

        num_ps = self.D
        scale = torch.exp(params[:,idx:(idx+num_ps)])
        idx += num_ps

        num_ps = self.D
        shift = params[:,idx:(idx+num_ps)]
        idx += num_ps

        scale = scale[:,None,:] 
        shift = shift[:,None,:] 

        z = (z - shift)/scale
        log_det = torch.sum(torch.log(scale), axis=2)

        return z, log_det

    def count_num_params(self,):
        return 2*self.D

class BatchNorm(Bijector):
    """Batch Norm bijector.

    For normalizing flows, it's useful to have a BatchNorm layer that
    propagates its log-determinant jacobian.  This Bijector keeps track of its
    most recently used normalization parameters, which can be invoked in the
    forward transform when use_last=True.

    :param D: Dimensionality of the bijection.
    :type D: int
    :param momentum: Momentum parameter of batch norm, default 0.1.
    :type momentum: float, optional.
    :param eps: Eps parameter of batch norm, default 1e-5.
    :type eps: float, optional.
    """
    def __init__(self, D, momentum=0.1, eps=1e-5):
        super().__init__(D)
        self.name = "BatchNorm"
        self.momentum = momentum
        self.eps = eps
        self.batch_norm = torch.nn.BatchNorm1d(
            D, eps=self.eps, momentum=momentum, affine=False
        )
        self.__last_mean = torch.tensor(np.zeros(D)).float()
        self.__last_alpha = torch.tensor(np.ones(D)).float()

    @property
    def momentum(self,):
        return self.__momentum

    @momentum.setter
    def momentum(self, val):
        if type(val) is not float:
            raise (TypeError(format_type_err_msg(self, "momentum", val, float)))
        elif val < 0.0:
            raise (ValueError("BatchNorm.momentum cannot be negative."))
        elif val > 1.0:
            print(
                "Warning: BathNorm.momentum  set to maximum of 1.0 (received %.2E)."
                % val
            )
            self.__momentum = 1.0
        else:
            self.__momentum = val

    @property
    def eps(self,):
        return self.__eps

    @eps.setter
    def eps(self, val):
        if type(val) is not float:
            raise (TypeError(format_type_err_msg(self, "eps", val, float)))
        elif val < 0.0:
            raise (ValueError("BatchNorm.eps cannot be negative."))
        else:
            self.__eps = val

    def get_last_mean(self,):
        return self.__last_mean

    def get_last_alpha(self,):
        return self.__last_alpha

    def __call__(self, z, use_last=False):
        return self.forward_and_log_det(z, use_last=use_last)

    def forward_and_log_det(self, z, use_last=False):
        """Batch norm forward and log determinant of the jacobian.

        :param z: Input to the bijector (M, N, D).
        :type z: torch.tensor
        :param use_last: Use previous mean and alpha of batch norm, default False.
        :type params: bool, optional
        """
        if use_last:
            alpha = self.__last_alpha
            z_norm = (z - self.__last_mean) / alpha

        else:
            z_size = z.size()
            z_vec = z.view(-1, self.D)
            z_var = torch.var(z_vec, dim=0)

            z_norm = self.batch_norm(z_vec)
            z_norm_var = torch.var(z_norm, dim=0)
            alpha = torch.sqrt(z_var) / torch.sqrt(z_norm_var)
            zn_alpha = z_norm*alpha[None, :]
            mean = torch.mean(z_vec - zn_alpha, dim=0)

            z_norm = z_norm.view(z_size[0], z_size[1], self.D)

            self.__last_mean = mean
            self.__last_alpha = alpha

        log_det = -torch.sum(torch.log(alpha))
        return z_norm, log_det

    def inverse_and_log_det(self, z):
        alpha = self.__last_alpha
        mean = self.__last_mean
        z = z * alpha
        z = z + mean
        log_det = -torch.sum(torch.log(alpha))
        return z, log_det


class ToSimplex(Bijector):
    """Maps tensor in (M,N,D-1) to D-simplex.

    :param D: Dimensionality of the bijection.
    :type D: int
    """
    def __init__(self, D):
        super().__init__(D)
        self.name = "ToSimplex"

    def __call__(self, z):
        return self.forward_and_log_det(z)

    def forward_and_log_det(self, z):
        """Forward transform and log det jac of mapping to D-simplex.

        :param z: Input to the bijector (M, N, D).
        :type z: torch.tensor
        """
        EPS = 1e-10
        ex = torch.exp(z)
        sum_ex = torch.sum(ex, dim=2)
        den = sum_ex + 1.0
        log_det = (
            torch.log(1.0 - (sum_ex / den) + EPS)
            - self.D * torch.log(den)
            + torch.sum(z, axis=2)
        )
        z = torch.cat((ex / den[:, :, None], 1.0 / den[:, :, None]), axis=2)

        return z, log_det

def dbg_check(tensor, name):
    num_elems = 1
    for dim in tensor.shape:
        num_elems *= dim
    num_infs = torch.sum(torch.isinf(tensor)).item()
    num_nans = torch.sum(torch.isnan(tensor)).item()

    print(name, "infs %d/%d" % (num_infs, num_elems), "nans %d/%d" % (num_nans, num_elems))
    return None
