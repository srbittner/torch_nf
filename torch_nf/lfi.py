import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_nf.util import plot_dist
from torch_nf.error_formatters import dbg_check
import torch_nf.density_estimator as de
import time

def train_APT(
    cde, system, x0, M=500, M_atom=100, R=10, num_iters=1000, z0=None, verbose=True
):
    x0_torch = torch.tensor(x0).float()
    has_norm_flow = (type(cde.density_estimator) == de.NormFlow)
    has_MoG = (type(cde.density_estimator) == de.MoG)
    losses = []
    sample_times = []

    # Plot posterior at the beginning of optimization.
    if verbose:
        z, q_prop, x = SNPE_proposal(2, M, system, cde, x0_torch)
        plt.figure()
        if cde.density_estimator.D > 8:
            plot_inds = [i for i in range(8)]
        else:
            plot_inds = None
        plot_dist(
            z.numpy(),
            np.log(q_prop.numpy()+1e-20),
            z0=z0,
            z_labels=system.z_labels,
            inds=plot_inds,
            lb=system.lb,
            ub=system.ub,
        )
        plt.show()

    zs = []
    log_probs = []
    optimizer = torch.optim.Adam(cde.parameters(), lr=1e-3)
    alphas, mus, Sigma_invs = [], [], []
    for r in range(1, R + 1):
        print('r', r)
        it_times = []
        time1 = time.time()
        z, q_prop, x = SNPE_proposal(r, M, system, cde, x0_torch)
        sample_times.append(time.time() - time1)
        log_q_prior = torch.tensor(system.prior.logpdf(z.numpy())).float()

        # APT with MoG needs gaussian proposal params.
        if has_MoG:
            params = cde.param_net(x)
            alpha_prop, mu_prop, Sigma_prop_inv, Sigma_prop_det = cde.density_estimator._get_MoG_params(params)
            alpha_prop = alpha_prop.detach()
            mu_prop = mu_prop.detach()
            Sigma_prop_inv = Sigma_prop_inv.detach()
            Sigma_prop_det = Sigma_prop_det.detach()

        if r == 1:
            x_all = x
            z_all = z
            if has_norm_flow:
                log_q_prior_all = log_q_prior

            zs.append(z.numpy())
            log_probs.append(log_q_prior.numpy())
        elif r==2 and has_MoG:
            x_all = x
            z_all = z
            alpha_prop_all = alpha_prop
            mu_prop_all = mu_prop
            Sigma_prop_inv_all =  Sigma_prop_inv
            Sigma_prop_det_all =  Sigma_prop_det
        else:
            x_all = torch.cat((x_all, x), dim=0)
            z_all = torch.cat((z_all, z), dim=0)
            log_q_prior_all = torch.cat((log_q_prior_all, log_q_prior), dim=0)

            if (has_MoG):
                alpha_prop_all = torch.cat((alpha_prop_all, alpha_prop), dim=0)
                mu_prop_all = torch.cat((mu_prop_all, mu_prop), dim=0)
                Sigma_prop_inv_all = torch.cat(
                    (Sigma_prop_inv_all, Sigma_prop_inv), 
                    dim=0
                )
                Sigma_prop_det_all = torch.cat(
                    (Sigma_prop_det_all, Sigma_prop_det),
                    cim=0
                )

        # Mini-batching for norm flows.
        if has_norm_flow:
            M_batch = M * r
            batch_buf = np.random.permutation(M_batch)
            j = 0
        if has_MoG:
            z = z_all
            x = x_all
            if r > 1:
                alpha_prop = alpha_prop_all
                mu_prop = mu_prop_all
                Sigma_prop_inv = Sigma_prop_inv_all
                Sigma_prop_det = Sigma_prop_det_all

        for i in range(1, num_iters + 1):
            #print(42*'*')
            #print('r', r, 'i', i)
            #print(42*'*')
            if has_norm_flow:
                if r==1:
                    z = z_all
                    x = x_all
                    log_q_prior = log_q_prior_all
                else:
                    if M_batch - j < M_atom:
                        batch_buf = np.random.permutation(M_batch)
                        j = 0
                    batch_inds = batch_buf[j : (j + M_atom)]
                    j += M_atom
                    z = z_all[batch_inds]
                    x = x_all[batch_inds]
                    log_q_prior = log_q_prior_all[batch_inds]
                

            time1 = time.time()

            if has_norm_flow:
                # update the batch norm
                _, _ = cde(x, N=1)
                if (r==1):
                    log_prob = cde.log_prob(z[:,None,:], x)
                    log_q_tilde = log_prob[:,0]
                else:
                    z_in = z[None, :, :].repeat(M_atom, 1, 1)
                    log_prob = cde.log_prob(z_in, x)
                    log_num = torch.diag(log_prob) - log_q_prior
                    log_q_div_p = log_prob - log_q_prior[None, :]
                    log_denom = torch.logsumexp(log_q_div_p, axis=1)
                    log_q_tilde = log_num - log_denom
            if has_MoG:
                if (r==1):
                    params = cde.param_net(x)
                    #dbg_check(params, 'params')
                    alpha, mu, Sigma_inv, Sigma_det = cde.density_estimator._get_MoG_params(params)
                    alphas.append(alpha.detach().numpy())
                    mus.append(mu.detach().numpy())
                    Sigma_invs.append(Sigma_inv.detach().numpy())

                    log_prob = cde.log_prob(z[:,None,:], x)

                    log_q_tilde = log_prob[:,0]
                else:
                    params = cde.param_net(x)
                    #dbg_check(params, 'params')
                    alpha, mu, Sigma_inv, Sigma_det = cde.density_estimator._get_MoG_params(params)
                    alphas.append(alpha.detach().numpy())
                    mus.append(mu.detach().numpy())
                    Sigma_invs.append(Sigma_inv.detach().numpy())


                    mu0 = torch.zeros((system.D,)).float()
                    Sigma0_inv = torch.zeros((system.D, system.D)).float()
                    log_q_tilde = MoG_proposal_posterior(
                        z,
                        mu0,
                        Sigma0_inv,
                        alpha,
                        mu,
                        Sigma_inv,
                        Sigma_det,
                        alpha_prop,
                        mu_prop,
                        Sigma_prop_inv,
                        Sigma_prop_det,
                    )
                    #dbg_check(log_q_tilde, 'log_q_tilde')

            loss = -torch.mean(log_q_tilde)
            #dbg_check(loss, 'loss')
            _loss = loss.item()
            #print('loss', _loss)
            if (np.isnan(_loss)):
                alphas = np.array(alphas)
                K = alphas.shape[2]
                mean_alphas = np.mean(alphas, axis=1)
                std_alphas = np.std(alphas, axis=1)
                its = np.arange(alphas.shape[0])

                plt.figure()
                for k in range(mean_alphas.shape[1]):
                    plt.errorbar(its, mean_alphas[:,k], std_alphas[:,k])
                plt.title('alpha')
                plt.show()


                mus = np.array(mus)
                D = mus.shape[3]
                mean_mus = np.mean(mus, axis=1)
                std_mus = np.std(mus, axis=1)
                for d in range(D):
                    plt.figure()
                    for k in range(K):
                        plt.errorbar(its, mean_mus[:,k,d], std_mus[:,k,d])
                    plt.title('mu d=%d' % (d+1))
                    plt.show()

                Sigma_invs = np.array(Sigma_invs)
                mean_Sigma_invs = np.mean(Sigma_invs, axis=1)
                std_Sigma_invs = np.std(Sigma_invs, axis=1)
                for d in range(D):
                    plt.figure()
                    for k in range(K):
                        plt.errorbar(its, mean_Sigma_invs[:,k,d,d], std_Sigma_invs[:,k,d,d])
                    plt.title('Sigma_inv[d,d], d=%d' % (d+1))
                    plt.show()

                for d in range(D-1):
                    plt.figure()
                    for k in range(K):
                        plt.errorbar(its, mean_Sigma_invs[:,k,d,d+1], std_Sigma_invs[:,k,d,d+1])
                    plt.title('Sigma_inv[d,d+1], d=%d' % (d+1))
                    plt.show()
                break

                break

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #for ii, param in enumerate(cde.parameters()):
            #    dbg_check(param, 'param %d' % ii)
            #for ii, param in enumerate(cde.parameters()):
                #dbg_check(param.grad, 'param grad %d' % ii)
                #print(torch.sum(param.grad**2))
            torch.nn.utils.clip_grad_norm_(cde.parameters(), 0.1, 2)
            optimizer.step()

            if (r == 1 and i == 1) or i % (num_iters // 20) == 0:
                time2 = time.time()
                it_time = time2 - time1
                if i % (num_iters // 20) == 0:
                    it_times.append(it_time)
                print(
                    "r %d, it %d, loss=%.2E, time/it=%.3fs" % (r, i, _loss, it_time),
                    flush=True,
                )
                if verbose:
                    if r != 1 and (i % num_iters) == 0:
                        plt.figure()
                        plt.plot(-np.array(losses))
                        plt.ylim([-losses[100], -min(losses)])
                        plt.show()
            losses.append(loss.item())

        z, q_prop, x = SNPE_proposal(r + 1, M, system, cde, x0_torch)
        log_q_prop = np.log(q_prop.numpy()+1e-20)
        zs.append(z.numpy())
        log_probs.append(log_q_prop)
        if verbose:
            plt.figure()
            plot_dist(
                z.numpy(), log_q_prop, z0=z0, z_labels=system.z_labels, inds=plot_inds, lb=system.lb, ub=system.ub
            )
            plt.show()

    it_time = np.mean(np.array(it_times))
    losses = np.array(losses)
    zs = np.array(zs)
    log_probs = np.array(log_probs)
    sample_times = np.array(sample_times)
    return cde, losses, zs, log_probs, it_time, sample_times

def MoG_proposal_posterior(
    z, 
    mu0,
    Sigma0_inv,
    alpha,
    mu,
    Sigma_inv,
    Sigma_det,
    alpha_prop,
    mu_prop,
    Sigma_prop_inv,
    Sigma_prop_det,
):
    D = mu0.shape[0]

    _z = z[:,None,None,:,None]

    _alpha = alpha[:,:,None]
    _alpha_prop = alpha_prop[:,None,:]

    _mu = mu[:,:,None,:,None]
    _mu_prop = mu_prop[:,None,:,:,None]
    _mu0 = mu0[None,None,None,:,None]

    _Sigma_inv = Sigma_inv[:,:,None,:,:]
    _Sigma_prop_inv = Sigma_prop_inv[:,None,:,:,:]
    _Sigma0_inv = Sigma0_inv[None,None,None,:,:]

    # (M, K, K, D, D)
    Sigma_star_inv =  _Sigma_inv + _Sigma_prop_inv - _Sigma0_inv
    Sigma_star =  torch.inverse(Sigma_star_inv)

    mu_star = torch.matmul(
        Sigma_star,
        torch.matmul(_Sigma_inv, _mu) + \
        torch.matmul(_Sigma_prop_inv, _mu_prop) - \
        torch.matmul(_Sigma0_inv, _mu0)
    )

    mu_starT = torch.transpose(mu_star, 4, 3)
    _muT = torch.transpose(_mu, 4, 3)
    _mu_propT = torch.transpose(_mu_prop, 4, 3)

    exponent = torch.matmul(torch.matmul(mu_starT, Sigma_star_inv), mu_star) + \
        torch.matmul(torch.matmul(_muT, _Sigma_inv), _mu) + \
        torch.matmul(torch.matmul(_mu_propT, _Sigma_prop_inv), _mu_prop)
    exp_factor = torch.exp(-0.5*exponent[:,:,:,0,0])

    Sigma_star_det = 1. / torch.det(Sigma_star_inv)
    #det_Sigma = 1. / torch.det(_Sigma_inv)
    #det_Sigma_prop = 1. / torch.det(_Sigma_prop_inv)
    Sigma_det = Sigma_det[:,:,None]
    Sigma_prop_det = Sigma_prop_det[:,None,:]
    """
    print('min dets')
    print(torch.min(Sigma_star_det))
    print(torch.min(Sigma_det))
    print(torch.min(Sigma_prop_det))
    print('max dets')
    print(torch.max(Sigma_star_det))
    print(torch.max(Sigma_det))
    print(torch.max(Sigma_prop_det))
    """

    gamma = torch.sqrt(Sigma_star_det / (Sigma_det*Sigma_prop_det))
    gamma = gamma*exp_factor
    gamma = _alpha*_alpha_prop*gamma + 1e-12
    gamma = gamma / torch.sum(gamma, dim=(1,2), keepdim=True)

    z_mu = _z - mu_star
    z_mu_T = torch.transpose(z_mu, 4, 3)
    gauss_probs_num = torch.exp(-0.5*torch.matmul(torch.matmul(z_mu_T, Sigma_star_inv), z_mu))[:,:,:,0,0]
    gauss_probs_denom = torch.sqrt(((2*np.pi)**D)*Sigma_star_det)
    print('num', 'denom')
    print(gauss_probs_num.shape, gauss_probs_denom.shape)
    gauss_probs = gauss_probs_num / gauss_probs_denom

    log_q_tilde = torch.sum(gamma*gauss_probs, dim=(1,2))
    return log_q_tilde


def clip_grads(params, clip):
    for param in params():
        param.grad.data.clamp_(-clip, clip)
    return None


def SNPE_proposal(r, M, system, cde, x0):
    _M = 0
    while _M < M:
        if r == 1:
            _z = system.prior.rvs(M)
            _q_z = system.prior.pdf(_z)
        else:
            _z, _log_q_z = cde(x=x0, N=M, freeze_bn=True)
            _z = _z.detach().numpy()[0]
            _q_z = np.exp(_log_q_z.detach().numpy())[0]

        valid_inds = system.valid_samples(_z)
        _x = system.simulate(_z)
        _z = _z[valid_inds]
        _q_z = _q_z[valid_inds]
        _x = _x[valid_inds]
        if _M == 0:
            z = _z
            x = _x
            q_z = _q_z
        else:
            z = np.concatenate((z, _z), axis=0)
            q_z = np.concatenate((q_z, _q_z), axis=0)
            x = np.concatenate((x, _x), axis=0)
        _M += sum(valid_inds)
        print("Cumulative valid proposals:", _M)

    z = z[:M]
    q_z = q_z[:M]
    x = x[:M]

    z = torch.tensor(z).float()
    q_z = torch.tensor(q_z).float()
    x = torch.tensor(x).float()
    return z, q_z, x


def ABC_MCMC(N, system, proposal, T_x0, eps):
    count = 0
    z_last = system.prior.rvs(1)
    zs = []
    T_xs = []
    n_sims = 0
    time0 = time.time()
    while count < N:
        z = proposal.rvs(z_last)
        T_x = system.simulate(z)
        abc_accept = system.abc_accept(T_x, T_x0, eps)
        if abc_accept:
            log_p_z = system.prior.logpdf(z)
            log_p_z_last = system.prior.logpdf(z_last)
            log_q_z_z_last = proposal.logpdf(z, z_last[0,:])
            log_q_z_last_z = proposal.logpdf(z_last, z[0,:])
            log_mh_ratio = log_p_z + log_q_z_last_z - log_p_z_last - log_q_z_z_last
            if (log_mh_ratio < 0):
                alpha = np.exp(log_mh_ratio)
                if np.random.uniform(0., 1.) < alpha:
                    zs.append(z[0])
                    T_xs.append(T_x[0])
                    z_last = z
                    count += 1
            else:
                zs.append(z[0])
                T_xs.append(T_x[0])
                z_last = z
                count += 1
        else:
            if (count == 0 and time.time() - time0 > 10):
                return None, False
        n_sims += 1
        print('count=%d\r' % count, end="")
    return np.array(zs), True

def ABC_SMC(N, system, proposal, T_x0, all_eps, count_tol=1e6):
    T = all_eps.shape[0]
    z_last = system.prior.rvs(N)
    zs = [z_last]
    T_xs = [system.simulate(z_last)]
    n_sims = 0
    for t in range(T):
        eps = all_eps[t]
        z_t = []
        T_x_t = []
        for i in range(N):
            count = 0
            while(True):
                z_i = proposal.rvs(z_last[i])
                T_x = system.simulate(z_i[None, :])
                abc_accept = system.abc_accept(T_x, T_x0, eps)
                if abc_accept:
                    z_t.append(z_i)
                    T_x_t.append(T_x[0])
                    break
                count += 1
                print('t=%d, i=%d, count=%7d\r' % (t, i, count), end="", flush=True)
                if (count > count_tol):
                    print('SMC failed after a million proposals.')
                    return None
        zs.append(np.array(z_t))
        T_xs.append(np.array(T_x_t))
        
    return np.array(zs)



def train_SNPE(cnf, system, x0, M=500, R=10, num_iters=1000, verbose=True, z0=None):
    x0_torch = torch.tensor(x0).float()
    losses = []
    if verbose:
        print("init")
        z, q_prop, x = SNPE_proposal(2, M, system, cnf, x0_torch)
        #dbg_check(z, "z")
        #dbg_check(q_prop, "q_prop")
        z = z.detach()
        q_prop = q_prop.detach()
        plt.figure()
        plot_dist(z.numpy(), np.log(q_prop.numpy()), z0=z0, z_labels=system.z_labels, lb=system.lb, ub=system.ub)
        plt.show()

    zs = []
    log_probs = []
    sample_times = []
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    for r in range(1, R + 1):
        print('r', r)
        it_times = []
        time1 = time.time()
        z, q_prop, x = SNPE_proposal(r, M, system, cnf, x0_torch)
        sample_times.append(time.time() - time1)
        z, q_prop = z.detach(), q_prop.detach()
        q_prior = torch.tensor(system.prior.pdf(z.numpy())).float()
        w = q_prior / q_prop
        w = w / torch.sum(w)
        print("q_prop", torch.min(q_prop), torch.max(q_prop))
        print("w", torch.min(w), torch.max(w))
        for i in range(1, num_iters + 1):
            time1 = time.time()
            _, _ = cnf(x, N=1)

            log_prob = cnf.log_prob(z[:, None, :], x)
            loss = -torch.mean(w * log_prob[:, 0])
            _loss = loss.item()
            if np.isnan(_loss):
                break
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 0.1, 2)
            optimizer.step()

            if (r == 1 and i == 1) or i % (num_iters // 20) == 0:
                time2 = time.time()
                it_time = time2 - time1
                if i % (num_iters // 20) == 0:
                    it_times.append(it_time)
                print(
                    "r %d, it %d, loss=%.2E, time/it=%.3f" % (r, i, _loss, it_time),
                    flush=True,
                )
                if verbose:
                    if r != 1 and (i % num_iters) == 0:
                        plt.figure()
                        plt.plot(-np.array(losses))
                        plt.ylim([-losses[100], -losses[-1]])
                        plt.show()
                if np.isnan(_loss):
                    break
            losses.append(loss.item())

        z, q_prop, x = SNPE_proposal(r + 1, M, system, cnf, x0_torch)
        z = z.detach().numpy()
        log_q_prop = np.log(q_prop.detach().numpy())
        zs.append(z)
        log_probs.append(log_q_prop)
        if verbose:
            plt.figure()
            plot_dist(z, log_q_prop, z0=z0, z_labels=system.z_labels, lb=system.lb, ub=system.ub)
            plt.show()

    it_time = np.mean(np.array(it_times))
    losses = np.array(losses)
    zs = np.array(zs)
    log_probs = np.array(log_probs)
    return cnf, losses, zs, log_probs, it_time, sample_times


