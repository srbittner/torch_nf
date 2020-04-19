import numpy as np
import scipy.stats
from torch_nf.mf_v1 import mean_field_EI, mean_field_4n


class System(object):
    def __init__(self, D):
        super(System, self).__init__()
        self.D = D
        self.z_labels = None
        self.support_layer = None
        
    def simulate(self,):
        raise NotImplementedError()
        
    def sample_prior(self, M):
        raise NotImplementedError()

    def abc_accept(self, x, x0, eps):
        rho = np.linalg.norm(x-x0, axis=1)
        return rho < eps

    def valid_samples(self, x):
        return np.ones((x.shape[0],), dtype=bool)
        
class Gauss(System):
    def __init__(self, D, N):
        super(Gauss, self).__init__(D)
        self.N = N
        self.mu_0 = np.zeros((D,))
        self.Sigma_0 = 2*np.eye(D)
        self.mvn_0 = scipy.stats.multivariate_normal(mean=self.mu_0, cov=self.Sigma_0)
        self.Sigma = np.eye(D)
        self.mvn = scipy.stats.multivariate_normal(mean=np.zeros(D,), cov=self.Sigma)
        self.prior = self.mvn_0
        
    def simulate(self, z):
        M = z.shape[0]
        mu = z
        eps = np.reshape(self.mvn.rvs(M*self.N), (M, self.N, self.D))
        x = np.expand_dims(mu, 1) + eps
        #x = np.reshape(x, (M, self.N*self.D))
        x = np.mean(x, axis=1)
        return x

    def valid_samples(self, x):
        return np.ones((x.shape[0],), dtype=bool)


class Mat(System):
    def __init__(self, d):
        self.d = d
        D = (d+1)*d//2
        super(Mat, self).__init__(D)
        self.lb = -np.ones((D,))
        self.ub = np.ones((D,))
        self.prior = Uniform(self.lb, self.ub)

    def simulate(self, z):
        M = z.shape[0]
        x = np.zeros((M, self.d, self.d))
        ind = 0
        for i in range(self.d):
            for j in range(i, self.d):
                if (i==j):
                    x[:,i,j] = z[:, ind]
                x[:, j,i] = z[:,ind]
                ind += 1
        e, v = np.linalg.eigh(x)
        det = np.prod(e, axis=1)
        trace = np.sum(e, axis=1)
        T_x = np.stack((det, trace), axis=1)
        return T_x

    def abc_accept(self, x, x0, eps):
        diff = x - x0
        return np.logical_and(np.abs(diff[:,0]) < eps[0], np.abs(diff[:,1]) < eps[1])

    def valid_samples(self, x):
        return np.ones((x.shape[0],), dtype=bool)

class Toy(System):
    def __init__(self, N):
        D = 5
        super(Toy, self).__init__(D)
        self.N = N
        self.D_x =2 
        self.lb = -3.*np.ones((D,))
        self.ub = 3.*np.ones((D,))
        self.prior = Uniform(self.lb, self.ub)

    def simulate(self, z):
        EPS = 1e-3
        M = z.shape[0]
        m = z[:,:2]
        s1 = z[:,2]**2
        s2 = z[:,3]**2
        rho = np.tanh(z[:,4])
        off_diag = rho*s1*s2
        Sigma = np.transpose(
            np.array([[s1**2+EPS, off_diag],
                      [off_diag, s2**2+EPS]]),
            (2, 0, 1)
        )

        x = np.zeros((M, self.D_x*self.N))
        for i in range(M):
            dist = scipy.stats.multivariate_normal(mean=m[i], cov=Sigma[i])
            x_i = dist.rvs(self.N)
            x[i,:] = np.reshape(x_i, (self.D_x*self.N,))
        return x
        
class MF_V1(System):
    def __init__(self,):
        D = 8
        super(MF_V1, self).__init__(D)
        self.lb = np.array([0., -20., 0., -20., 0., 0., 0., 0.])
        self.ub = np.array([20.,  0., 20.,  0., 1., 1., 1., 1.])
        self.prior = Uniform(self.lb, self.ub)
        self.z_labels = [r'$W_{EE}$', r'$W_{EI}$', r'$W_{IE}$', r'$W_{II}$', \
                         r'$\sigma_{EE}$', r'$\sigma_{EI}$', r'$\sigma_{IE}$', r'$\sigma_{II}$']

    def simulate(self, z):
        return mean_field_EI(z, traj=False)

class MF_V1_4n(System):
    def __init__(self, sigmas=False):
        if sigmas:
            D = 36
        else:
            D = 20
        super(MF_V1_4n, self).__init__(D)
        if sigmas:
            self.lb = 20.*np.array([0., -1., -1., -1.,
                                0., -1., -1., -1.,
                                0., -1., -1., -1.,
                                0., -1., -1., -1.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,])
            self.ub = np.array([20., 0., 0., 0.,
                                20., 0., 0., 0.,
                                20., 0., 0., 0.,
                                20., 0., 0., 0.,
                                2., 2., 2., 2.,
                                2., 2., 2., 2.,
                                2., 2., 2., 2.,
                                2., 2., 2., 2.])
        else:
            self.lb = 20.*np.array([0., -1., -1., -1.,
                                0., -1., -1., -1.,
                                0., -1., -1., -1.,
                                0., -1., -1., -1.])
            self.ub = 20.*np.array([1., 0., 0., 0.,
                                    1., 0., 0., 0.,
                                    1., 0., 0., 0.,
                                    1., 0., 0., 0.])
        self.lb = np.concatenate((self.lb, 0.*np.ones((4,))), axis=0)
        self.ub = np.concatenate((self.ub,  2.*np.ones((4,))), axis=0)

        self.prior = Uniform(self.lb, self.ub)
        self.z_labels = [r'$W_{EE}$', r'$W_{EP}$', r'$W_{ES}$', r'$W_{EV}$', \
                         r'$W_{PE}$', r'$W_{PP}$', r'$W_{PS}$', r'$W_{PV}$', \
                         r'$W_{SE}$', r'$W_{SP}$', r'$W_{SS}$', r'$W_{SV}$', \
                         r'$W_{VE}$', r'$W_{VP}$', r'$W_{VS}$', r'$W_{VV}$']
        self.T_x_labels = [r'$\mu_{E}(0)$', r'$\mu_{S}(0)$', r'$\mu_{V}(0)$',
                           r'$\mu_{E}(6)$', r'$\mu_{S}(6)$', r'$\mu_{V}(6)$',
                           r'$\mu_{E}(12)$', r'$\mu_{S}(12)$', r'$\mu_{V}(12)$',
                           r'$\Delta_{E}(0)$', r'$\Delta_{S}(0)$', r'$\Delta_{V}(0)$',
                           r'$\Delta_{E}(6)$', r'$\Delta_{S}(6)$', r'$\Delta_{V}(6)$',
                           r'$\Delta_{E}(12)$', r'$\Delta_{S}(12)$', r'$\Delta_{V}(12)$']
        if vars:
            self.z_labels += [r'$\sigma_{EE}$', r'$\sigma_{EP}$', r'$\sigma_{ES}$', r'$\sigma_{EV}$', \
                              r'$\sigma_{PE}$', r'$\sigma_{PP}$', r'$\sigma_{PS}$', r'$\sigma_{PV}$', \
                              r'$\sigma_{SE}$', r'$\sigma_{SP}$', r'$\sigma_{SS}$', r'$\sigma_{SV}$', \
                              r'$\sigma_{VE}$', r'$\sigma_{VP}$', r'$\sigma_{VS}$', r'$\sigma_{VV}$']

    def simulate(self, z):
        return mean_field_4n(z, traj=False)

    def valid_samples(self, x):
        return np.logical_and(0 < x, x < 1e3).all(axis=1)


class Uniform(object):
    def __init__(self, lb, ub):
        self.D = lb.shape[0]
        self.lb = lb
        self.ub = ub

    def rvs(self, M):
        z = np.zeros((M, self.D))
        for i in range(self.D):
            z[:,i] = np.random.uniform(self.lb[i], self.ub[i], (M,))
        return z

    def pdf(self, z):
        M = z.shape[0]
        p = 1./np.prod(self.ub - self.lb)
        return p*np.ones((M,))
        
    def logpdf(self, z):
        M = z.shape[0]
        logp = -np.sum(np.log(self.ub - self.lb))
        return logp*np.ones((M,))

class GaussianProposal(object):
    def __init__(self, Sigma, lb, ub):
        self.D = lb.shape[0]
        self.Sigma = Sigma
        self.lb = lb
        self.ub = ub
        self.L = np.linalg.cholesky(Sigma)


    def rvs(self, mu, M=1):
        zs = []
        count = 0
        while (count < M):
            omega = np.random.normal(0., 1., (self.D,))
            z = np.matmul(self.L, omega) + mu
            if np.logical_and(self.lb < z, z < self.ub).all():
                zs.append(z)
                count += 1
        return np.concatenate(zs, axis=0)

    def pdf(self, z, mu):
        dist = scipy.stats.multivariate_normal(mean=mu, cov=self.Sigma)
        return dist.pdf(z)
    
    def logpdf(self, z, mu):
        dist = scipy.stats.multivariate_normal(mean=mu, cov=self.Sigma)
        return dist.logpdf(z)



