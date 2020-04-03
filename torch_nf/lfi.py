import torch
import numpy as np
from torch_nf.util import plot_dist
import matplotlib.pyplot as plt

def train_SNPE(cnf, system, x0, M=500, R=10, num_iters=1000, verbose=True, z0=None):
    x0_torch = torch.tensor(x0).float()
    losses = []
    for r in range(1, R+1):
        optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
        z, q_prop = SNPE_proposal(r, M, system, cnf, x0_torch)
        if verbose:
            print(torch.mean(z, axis=0), torch.var(z, axis=0))
            print('z mean, var')
        z, q_prop = z.detach(), q_prop.detach()
        q_prior = torch.tensor(system.prior.pdf(z.numpy())).float()
        w = q_prior/q_prop
        w = w / torch.sum(w)
        print('w', torch.min(w), torch.max(w))
        x = system.simulate(z.numpy())
        x = torch.tensor(x).float()
        for i in range(1, num_iters+1):
            log_prob = cnf.log_prob(z[:,None,:], x)
            #dbg_check(log_prob, 'log_prob')
            loss = - torch.mean(w*log_prob[:,0])
            _loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            #for j, param in enumerate(cnf.parameters()):
            #    dbg_check(param.grad, 'param.grad %d' % (j+1))
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 0.1, 2)
            optimizer.step()

            if (verbose and ((r==1 and i==1) or i % (num_iters//5) == 0)):
                print('r %d, it %d, loss=%.2E' % (r, i, _loss))
                if (np.isnan(_loss)):
                    break
            losses.append(loss.item())
           
        if verbose:
            z, q_prop = SNPE_proposal(r+1, M, system, cnf, x0_torch)
            z = z.detach()
            q_prop = q_prop.detach()
            plt.figure()
            plot_dist(z.numpy(), q_prop.numpy(), z0=z0)
            plt.show()

    return cnf, losses

def clip_grads(params, clip):
    for param in params():
        param.grad.data.clamp_(-clip, clip)
    return None

def SNPE_proposal(r, M, system, cnf, x0):
    if (r==1):
        z = system.prior.rvs(M)
        q_z = system.prior.pdf(z)
        z = torch.tensor(z).float()
        q_z = torch.tensor(q_z).float()
        return z, q_z
    else:
        z, log_q_z = cnf(x=x0, N=M)
        return z[0], torch.exp(log_q_z[0])
