""" Exponential family classes for implementing EFN. """

import torch
import numpy as np
import scipy
from scipy.stats import invwishart
from torch_nf.bijectors import Bijector, ToSimplex
from torch_nf.error_formatters import format_type_err_msg

class ExponentialFamily(object):
    """ Exponential family distributions for training EFNs.

    In this class, eta is the natural parameter and T calculates the sufficient
    statistics.  Here, eta is the augmented natural parameter that has a 1
    tacked onto the end when the log base measure is not a constant w.r.t. z.
    Likewise, T tacks on log(h(z)) when it is not a constant for the 
    exponential family.

    :param D: Dimensionality of random variable.
    :type D: int
    """
    def __init__(self, D, support_layer=None):
        super().__init__()
        self.D = D
        self.support_layer = support_layer
        self.D_eta = self._get_D_eta()

    @property
    def D(self):
        return self.__D
    @D.setter
    def D(self, val):
        if type(val) is not int:
            raise(TypeError(format_type_err_msg(self, "D", val, int)))
        elif (val < 1):
            raise ValueError("Exponential family dimensionality must be greater than 1.")
        self.__D = val
   
    @property
    def support_layer(self):
        return self.__support_layer
    @support_layer.setter
    def support_layer(self, val):
        if (val is None or issubclass(val, Bijector)):
            self.__support_layer = val
        else:
            raise(TypeError(format_type_err_msg(self, "support_layer", val, Bijector)))
        self.__support_layer = val
            


    def _get_D_eta(self,):
        """Calculate dimensionality of natural parameter.

        :return: Dimensionality of natural parameter eta.
        :rtype: int
        """
        return self.D

    def sample_eta(self, N):
        """Sample from prior distribution of the natural parameter eta.

        :param N: Number of eta samples.
        :type N: int
        :return: Samples of eta (N, D_eta).
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def mu_to_eta(self, mu):
        """Obtain the natural parameters (eta) from the mean parameterization (mu).

        :param mu: Mean parameterization (N, D_mu).
        :type mu: np.ndarray
        :return: eta (N, D_eta).
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def eta_to_mu(self, eta):
        """Obtain the mean parameterization (mu) from the natural parameters (eta).

        :param eta: Mean parameterization (N, D_eta).
        :type eta: np.ndarray
        :return: mu (N, D_mu).
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def T(self, z):
        """Sufficient statistic of the exponential family.

        :param z: Random variable of the exponential family (M, N, D).
        :type z: torch.tensor
        :return: T(z) (M, N, D_eta).
        :rtype: torch.tensor
        """
        raise NotImplementedError()




class MVN(ExponentialFamily):
    def __init__(self, D):
        super().__init__(D, None)

    def _get_D_eta(self,):
        """Calculate dimensionality of natural parameter.

        :return: Dimensionality of natural parameter eta.
        :rtype: int
        """
        return int(self.D + (self.D * (self.D + 1) // 2))

    def sample_eta(self, N=50, sigma_mu=1., iw_df_fac=5):
        """Sample from prior distribution of the natural parameter eta.

        :math:`\\mu_i \\sim \\mathcal{N}(0, \\sigma_{\\mu})`
        :math:`\\Sigma \\sim \\mathcal{IW}(df=iw_df_fac*D, scale=df_fac*D*I)`

        :param N: Number of eta samples.
        :type N: int, optional
        :param sigma_mu: Standard deviation of Gaussian prior on :math:`\\mu`.
        :type sigma_mu: float, optional
        :param iw_df_fac: Degree of freedom multiplier.  df=iw_df_fac*D.
        :type iw_df_fac: int, optional
        :return: Samples of eta (N, D_eta).
        :rtype: np.ndarray
        """
        mu = np.random.normal(0.0, sigma_mu, (N, self.D))
        df = iw_df_fac * self.D
        iw = invwishart(df=df, scale=df * np.eye(self.D))
        Sigma = iw.rvs(N)
        if N == 1:
            Sigma = np.expand_dims(Sigma, 0)
        return self.mu_to_eta(mu, Sigma)

    def T(self, z):
        """Sufficient statistic of the exponential family.

        :math:`T(z) = z, zz^\top`
        
        We eliminate redundancy in T(z) by vectorizing the upper triangular
        part of the second moment matrix.

        :param z: Random variable of the exponential family (M, N, D).
        :type z: torch.tensor
        :return: T(z) (M, N, D_eta).
        :rtype: torch.tensor
        """
        utri0_inds = np.triu_indices(self.D, 0)
        zzT = torch.matmul(z[:, :, :, None], z[:, :, None, :])
        zzT = zzT[:, :, utri0_inds[0], utri0_inds[1]]
        return torch.cat((z, zzT), axis=2)

    def mu_to_eta(self, mu, Sigma):
        """Obtain the natural parameters (eta) from the mean parameterization (mu, Sigma).

        To avoid redundancy, we eliminate the copy of symmetric elements of the
        second moment natural parameter.  We multiply such off diagonal elems
        by two in their vectorization in eta.

        :param mu: Mean (N, D)
        :type mu: np.ndarray
        :param Sigma: Covariance (N, D, D).
        :type Sigma: np.ndarray
        :return: eta (N, D_eta).
        :rtype: np.ndarray
        """
        utri0_inds = np.triu_indices(self.D, 0)
        utri1_inds = np.triu_indices(self.D, 1)

        Sigma_inv = np.linalg.inv(Sigma)
        eta1 = np.float64(np.matmul(Sigma_inv, np.expand_dims(mu, 2)))
        eta2 = np.float64(-Sigma_inv / 2)
        # Since we're using the minimal representation, we need to multiply eta
        # by two for the off diagonal elements.
        eta2[:, utri1_inds[0], utri1_inds[1]] = (
            2 * eta2[:, utri1_inds[0], utri1_inds[1]]
        )
        eta2_minimal = eta2[:, utri0_inds[0], utri0_inds[1]]
        eta = np.concatenate((eta1[:, :, 0], eta2_minimal), axis=1)
        return eta

    def eta_to_mu(self, eta):
        """Obtain the mean parameterization (mu, Sigma) from the natural parameters (eta).

        :param eta: Mean parameterization (N, D_eta).
        :type eta: np.ndarray
        :return: mu (N, D), Sigma (N, D, D).
        :rtype: np.ndarray, np.ndarray
        """
        N = eta.shape[0]
        eta1 = eta[:, : self.D]
        _eta2 = eta[:, self.D :]
        eta2 = np.zeros((N, self.D, self.D))
        inds = np.triu_indices(self.D)
        eta2[:, inds[0], inds[1]] = _eta2
        eta2 = (eta2 + np.transpose(eta2, (0, 2, 1))) / 2
        eta2_inv = np.linalg.inv(eta2)
        mu = np.matmul(-0.5 * eta2_inv, np.expand_dims(eta1, 2))
        Sigma = -0.5 * eta2_inv
        return mu[:, :, 0], Sigma

    def KL(self, z, log_prob, eta):
        M = z.shape[0]
        KLs = np.zeros((M,))
        mu, Sigma = self.eta_to_mu(eta)
        for i in range(M):
            dist = scipy.stats.multivariate_normal(mean=mu[i], cov=Sigma[i])
            log_p_z = dist.logpdf(z[i])
            KLs[i] = np.mean(log_prob[i] - log_p_z)
        return KLs


class Dirichlet(ExponentialFamily):
    def __init__(self, D):
        super().__init__(D, ToSimplex)

    def _get_D_eta(self,):
        """Calculate dimensionality of natural parameter.

        :return: Dimensionality of natural parameter eta.
        :rtype: int
        """
        # We add one to the dimensionality of alpha for the log base measure.
        return self.D + 1

    def sample_eta(self, N=50, lb=0.5, ub=2.):
        """Sample from prior distribution of the natural parameter eta.

        :math:`\\alpha_i \\sim \\mathcal{U}U[lb, ub]`

        A one is tacked onto the end of this natural parameter because there
        is a non-constant log base measure for the Dirichlet.

        :param N: Number of eta samples.
        :type N: int, optional
        :param lb: Lower bound of prior.
        :type lb: float, optional
        :param ub: Upper bound of prior.
        :type ub: float, optional
        :return: Samples of eta (N, D_eta).
        :rtype: np.ndarray
        """
        alpha = np.random.uniform(lb, ub, (N, self.D))
        # need to tack on 1's for the base measure
        eta = np.concatenate((alpha, np.ones((N, 1))), axis=1)
        return eta

    def T(self, z):
        """Sufficient statistic of the exponential family.

        :math:`T(z) = \\log(z)
        :math:`\\log(h(z)) = \\sum_i \\log(z_i)

        The log base measure is concatenated onto the end of T(z).

        :param z: Random variable of the exponential family (M, N, D).
        :type z: torch.tensor
        :return: T(z) (M, N, D_eta).
        :rtype: torch.tensor
        """
        log_z = torch.log(z)
        h_z = torch.sum(log_z, dim=2, keepdim=True)
        return torch.cat((log_z, h_z), axis=2)

    def mu_to_eta(self, alpha):
        """Obtain the natural parameters (eta) from the mean parameterization (alpha).

        :param alpha: Mean parameterization (N, D_mu).
        :type alpha: np.ndarray
        :return: eta (N, D_eta).
        :rtype: np.ndarray
        """
        N = alpha.shape[0]
        eta = np.concatenate((alpha, np.ones((N, 1))), axis=1)
        return eta

    def eta_to_mu(self, eta):
        """Obtain the mean parameterization (alpha) from the natural parameters (eta).

        :param eta: Mean parameterization (N, D_eta).
        :type eta: np.ndarray
        :return: mu (N, D_mu).
        :rtype: np.ndarray
        """
        alpha = eta[:, : self.D]
        return alpha

    def KL(self, z, log_prob, eta):
        M = z.shape[0]
        N = z.shape[1]
        KLs = np.zeros((M,))
        alpha = self.eta_to_mu(eta)
        simplex_eps = 1e-32
        for i in range(M):
            dist = scipy.stats.dirichlet(alpha=np.float64(alpha[i]))
            zi = np.float64(z[i]) + simplex_eps
            zi = zi / np.expand_dims(np.sum(zi, 1), 1)
            log_p_z = dist.logpdf(zi.T)
            KLs[i] = np.mean(log_prob[i] - log_p_z)
        return KLs

