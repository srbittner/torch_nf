import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_nf.util import plot_dist
from torch_nf.error_formatters import dbg_check
import time


def train_SNPE(cnf, system, x0, M=500, R=10, num_iters=1000, verbose=True, z0=None):
    x0_torch = torch.tensor(x0).float()
    losses = []
    if verbose:
        print("init")
        z, q_prop = SNPE_proposal(2, M, system, cnf, x0_torch)
        dbg_check(z, "z")
        dbg_check(q_prop, "q_prop")
        z = z.detach()
        q_prop = q_prop.detach()
        plt.figure()
        plot_dist(z.numpy(), np.log(q_prop.numpy()), z0=z0, z_labels=system.z_labels)
        plt.show()

    zs = []
    log_probs = []
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    for r in range(1, R + 1):
        print('r', r)
        it_times = []
        z, q_prop = SNPE_proposal(r, M, system, cnf, x0_torch)
        z, q_prop = z.detach(), q_prop.detach()
        q_prior = torch.tensor(system.prior.pdf(z.numpy())).float()
        w = q_prior / q_prop
        w = w / torch.sum(w)
        print("q_prop", torch.min(q_prop), torch.max(q_prop))
        print("w", torch.min(w), torch.max(w))
        x = system.simulate(z.numpy())
        x = torch.tensor(x).float()
        for i in range(1, num_iters + 1):
            # update the batch norm
            time1 = time.time()
            _, _ = cnf(x, N=1)

            log_prob = cnf.log_prob(z[:, None, :], x)
            # dbg_check(log_prob, 'log_prob')
            loss = -torch.mean(w * log_prob[:, 0])
            _loss = loss.item()
            if np.isnan(_loss):
                break
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # for j, param in enumerate(cnf.parameters()):
            #    dbg_check(param.grad, 'param.grad %d' % (j+1))
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

        z, q_prop = SNPE_proposal(r + 1, M, system, cnf, x0_torch)
        # dbg_check(z, 'z')
        # dbg_check(q_prop, 'q_prop')
        z = z.detach().numpy()
        log_q_prop = np.log(q_prop.detach().numpy())
        zs.append(z)
        log_probs.append(log_q_prop)
        if verbose:
            plt.figure()
            plot_dist(z, log_q_prop, z0=z0, z_labels=system.z_labels)
            plt.show()

    it_time = np.mean(np.array(it_times))
    losses = np.array(losses)
    zs = np.array(zs)
    log_probs = np.array(log_probs)
    return cnf, losses, zs, log_probs, it_time


def train_APT(
    cnf, system, x0, M=500, M_atom=100, R=10, num_iters=1000, z0=None, verbose=True
):
    x0_torch = torch.tensor(x0).float()
    losses = []
    sample_times = []
    if verbose:
        z, q_prop, x = SNPE_proposal(2, M, system, cnf, x0_torch)
        plt.figure()
        if cnf.nf.D > 8:
            plot_inds = [i for i in range(8)]
        else:
            plot_inds = None
        plot_dist(
            z.numpy(),
            np.log(q_prop.numpy()+1e-20),
            z0=z0,
            z_labels=system.z_labels,
            inds=plot_inds,
        )
        plt.show()

    zs = []
    log_probs = []
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    for r in range(1, R + 1):
        print('r', r)
        it_times = []
        time1 = time.time()
        z, q_prop, x = SNPE_proposal(r, M, system, cnf, x0_torch)
        sample_times.append(time.time() - time1)
        log_q_prior = torch.tensor(system.prior.logpdf(z.numpy())).float()

        if r == 1:
            zs.append(z.numpy())
            x_all = x
            z_all = z
            log_q_prior_all = log_q_prior
        else:
            x_all = torch.cat((x_all, x), dim=0)
            z_all = torch.cat((z_all, z), dim=0)
            log_q_prior_all = torch.cat((log_q_prior_all, log_q_prior), dim=0)
        M_batch = M * r
        batch_buf = np.random.permutation(M_batch)
        j = 0
        for i in range(1, num_iters + 1):
            if M_batch - j < M_atom:
                batch_buf = np.random.permutation(M_batch)
                j = 0
            batch_inds = batch_buf[j : (j + M_atom)]
            j += M_atom
            z = z_all[batch_inds]
            x = x_all[batch_inds]
            log_q_prior = log_q_prior_all[batch_inds]

            # update the batch norm
            time1 = time.time()
            _, _ = cnf(x, N=1)

            z_in = z[None, :, :].repeat(M_atom, 1, 1)
            log_prob = cnf.log_prob(z_in, x)
            log_num = torch.diag(log_prob) - log_q_prior
            log_q_div_p = log_prob - log_q_prior[None, :]
            log_denom = torch.logsumexp(log_q_div_p, axis=1)
            log_q_tilde = log_num - log_denom
            loss = -torch.mean(log_q_tilde)
            _loss = loss.item()

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

        z, q_prop, x = SNPE_proposal(r + 1, M, system, cnf, x0_torch)
        log_q_prop = np.log(q_prop.numpy()+1e-20)
        zs.append(z.numpy())
        log_probs.append(log_q_prop)
        if verbose:
            plt.figure()
            plot_dist(
                z.numpy(), log_q_prop, z0=z0, z_labels=system.z_labels, inds=plot_inds
            )
            plt.show()

    it_time = np.mean(np.array(it_times))
    losses = np.array(losses)
    zs = np.array(zs)
    log_probs = np.array(log_probs)
    sample_times = np.array(sample_times)
    return cnf, losses, zs, log_probs, it_time, sample_times


def clip_grads(params, clip):
    for param in params():
        param.grad.data.clamp_(-clip, clip)
    return None


def SNPE_proposal(r, M, system, cnf, x0):
    _M = 0
    while _M < M:
        if r == 1:
            _z = system.prior.rvs(M)
            _q_z = system.prior.pdf(_z)
        else:
            _z, _log_q_z = cnf(x=x0, N=M, freeze_bn=True)
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
