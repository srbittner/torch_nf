import torch
import numpy as np
import scipy
from scipy.stats import invwishart
from torch_nf.bijectors import ToSimplex


class ExponentialFamily(object):
    def __init__(self, D):
        super().__init__()
        self.D = D
        self.support_layer = None
        self.D_eta = self._get_D_eta()

    def _get_D_eta(self,):
        return self.D

    def sample_eta(self, N):
        raise NotImplementedError()

    def mu_to_eta(self, mu):
        raise NotImplementedError()

    def eta_to_mu(self, eta):
        raise NotImplementedError()

    def T(self, z):
        raise NotImplementedError()


class Dirichlet(ExponentialFamily):
    def __init__(self, D):
        super().__init__(D)
        self.support_layer = ToSimplex

    def _get_D_eta(self,):
        return self.D + 1

    def sample_eta(self, N=50):
        alpha = np.random.uniform(0.5, 2.0, (N, self.D))
        # need to tack on 1's for the base measure
        eta = np.concatenate((alpha, np.ones((N, 1))), axis=1)
        return eta

    def T(self, z):
        # The suff stats
        log_z = torch.log(z)
        # Need to tack on the base measure as well
        h_z = torch.sum(log_z, dim=2, keepdim=True)
        return torch.cat((log_z, h_z), axis=2)

    def mu_to_eta(self, alpha):
        N = alpha.shape[0]
        eta = np.concatenate((alpha, np.ones((N, 1))), axis=1)
        return eta

    def eta_to_mu(self, eta):
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


class MVN(ExponentialFamily):
    def __init__(self, D):
        super().__init__(D)

    def _get_D_eta(self,):
        return int(self.D + (self.D * (self.D + 1) // 2))

    def sample_eta(self, N=50):
        df_fac = 5
        mu = np.random.normal(0.0, 1.0, (N, self.D))
        df = df_fac * self.D
        iw = invwishart(df=df, scale=df * np.eye(self.D))
        Sigma = iw.rvs(N)
        if N == 1:
            Sigma = np.expand_dims(Sigma, 0)
        return self.mu_to_eta(mu, Sigma)

    def T(self, z):
        utri0_inds = np.triu_indices(self.D, 0)
        zzT = torch.matmul(z[:, :, :, None], z[:, :, None, :])
        zzT = zzT[:, :, utri0_inds[0], utri0_inds[1]]
        return torch.cat((z, zzT), axis=2)

    def mu_to_eta(self, mu, Sigma):
        utri0_inds = np.triu_indices(self.D, 0)
        utri1_inds = np.triu_indices(self.D, 1)

        Sigma_inv = np.linalg.inv(Sigma)
        eta1 = np.float64(np.matmul(Sigma_inv, np.expand_dims(mu, 2)))
        eta2 = np.float64(-Sigma_inv / 2)
        # by using the minimal representation, we need to multiply eta by two
        # for the off diagonal elements
        eta2[:, utri1_inds[0], utri1_inds[1]] = (
            2 * eta2[:, utri1_inds[0], utri1_inds[1]]
        )
        eta2_minimal = eta2[:, utri0_inds[0], utri0_inds[1]]
        eta = np.concatenate((eta1[:, :, 0], eta2_minimal), axis=1)
        return eta

    def eta_to_mu(self, eta):
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
