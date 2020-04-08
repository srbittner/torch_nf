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
        dbg_check(z, 'z')
        dbg_check(q_prop, 'q_prop')
        z = z.detach()
        q_prop = q_prop.detach()
        plt.figure()
        plot_dist(z.numpy(), np.log(q_prop.numpy()), z0=z0, z_labels=system.z_labels)
        plt.show()

    zs = []
    log_probs = []
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    for r in range(1, R + 1):
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
            #dbg_check(log_prob, 'log_prob')
            loss = -torch.mean(w * log_prob[:, 0])
            _loss = loss.item()
            if np.isnan(_loss):
                break
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            #for j, param in enumerate(cnf.parameters()):
            #    dbg_check(param.grad, 'param.grad %d' % (j+1))
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 0.1, 2)
            optimizer.step()

            if ((r == 1 and i == 1) or i % (num_iters // 20) == 0):
                time2 = time.time()
                it_time = time2-time1
                if i % (num_iters // 20) == 0:
                    it_times.append(it_time)
                print("r %d, it %d, loss=%.2E, time/it=%.3f" % (r, i, _loss, it_time), flush=True)
                if verbose:
                    if (r != 1 and (i % num_iters) == 0):
                        plt.figure()
                        plt.plot(-np.array(losses))
                        plt.ylim([-losses[100], -losses[-1]])
                        plt.show()
                if np.isnan(_loss):
                    break
            losses.append(loss.item())

        z, q_prop = SNPE_proposal(r + 1, M, system, cnf, x0_torch)
        #dbg_check(z, 'z')
        #dbg_check(q_prop, 'q_prop')
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


def train_APT(cnf, system, x0, M=500, M_atom=100, R=10, num_iters=1000, z0=None, verbose=True):
    x0_torch = torch.tensor(x0).float()
    losses = []
    if verbose:
        print("init")
        z, q_prop = SNPE_proposal(2, M, system, cnf, x0_torch)
        z = z.detach()
        q_prop = q_prop.detach()
        plt.figure()
        plot_dist(z.numpy(), np.log(q_prop.numpy()), z0=z0, z_labels=system.z_labels)
        plt.show()

    zs = []
    log_probs = []
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    for r in range(1, R + 1):
        it_times = []
        z, q_prop = SNPE_proposal(r, M, system, cnf, x0_torch)
        z, q_prop = z.detach(), q_prop.detach()
        q_prior = torch.tensor(system.prior.pdf(z.numpy())).float()
        x = system.simulate(z.numpy())
        x = torch.tensor(x).float()

        if (r==1):
            x_all = x
            z_all = z
            q_prior_all = q_prior
        else:
            x_all = torch.cat((x_all, x), dim=0)
            z_all = torch.cat((z_all, z), dim=0)
            q_prior_all = torch.cat((q_prior_all, q_prior), dim=0)
        M_batch = M*r
        batch_buf = np.random.permutation(M_batch)
        j = 0
        for i in range(1, num_iters + 1):
            if (M_batch - j < M_atom):
                batch_buf = np.random.permutation(M_batch)
                j = 0
            batch_inds = batch_buf[j:(j+M_atom)]
            j += M_atom
            z = z_all[batch_inds]
            x = x_all[batch_inds]
            q_prior = q_prior_all[batch_inds]

            # update the batch norm
            time1 = time.time()
            _, _ = cnf(x, N=1)
            
            z_in = z[None,:,:].repeat(M_atom, 1, 1)
            log_prob = cnf.log_prob(z_in, x)
            log_num = torch.diag(log_prob) - torch.log(q_prior)
            q_div_p = torch.exp(log_prob) / q_prior[None, :]
            denom = torch.sum(q_div_p, axis=1)
            log_denom = torch.log(denom)
            log_q_tilde = log_num - log_denom
            loss = -torch.mean(log_q_tilde)
            _loss = loss.item()
            if np.isnan(_loss):
                print("r %d, it %d, loss=%.2E" % (r, i, _loss), flush=True)
                break

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 0.1, 2)
            optimizer.step()

            if ((r == 1 and i == 1) or i % (num_iters // 20) == 0):
                time2 = time.time()
                it_time = time2-time1
                if i % (num_iters // 20) == 0:
                    it_times.append(it_time)
                print("r %d, it %d, loss=%.2E, time/it=%.3f" % (r, i, _loss, it_time), flush=True)
                if verbose:
                    if (r != 1 and (i % num_iters) == 0):
                        plt.figure()
                        plt.plot(-np.array(losses))
                        plt.ylim([-losses[100], np.max(-losses)])
                        plt.show()
            losses.append(loss.item())

        z, q_prop = SNPE_proposal(r + 1, M, system, cnf, x0_torch)
        z = z.detach().numpy()
        log_q_prop = np.log(q_prop.detach().numpy())
        zs.append(z)
        log_probs.append(log_q_prop)
        if verbose:
            plt.figure()
            plot_dist(z, log_q_prop, z0=z0, z_labels=system.z_labels)
            plt.show()

    it_time = np.mean(np.array(it_times))
    print(it_time)
    losses = np.array(losses)
    zs = np.array(zs)
    log_probs = np.array(log_probs)
    return cnf, losses, zs, log_probs, it_time


def clip_grads(params, clip):
    for param in params():
        param.grad.data.clamp_(-clip, clip)
    return None


def SNPE_proposal(r, M, system, cnf, x0):
    if r == 1:
        z = system.prior.rvs(M)
        q_z = system.prior.pdf(z)
        z = torch.tensor(z).float()
        q_z = torch.tensor(q_z).float()
        return z, q_z
    else:
        z, log_q_z = cnf(x=x0, N=M, freeze_bn=True)
        return z[0], torch.exp(log_q_z[0])
