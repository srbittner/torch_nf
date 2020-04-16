import numpy as np
from torch_nf.systems import Mat, GaussianProposal
from torch_nf.lfi import ABC_SMC
import scipy.stats
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=2)
parser.add_argument("--T", type=int, default=50)
parser.add_argument("--sigma", type=float, default=0.25)
parser.add_argument("--rs", type=int, default=1)

args = parser.parse_args()
d = args.d
T = args.T
sigma = args.sigma
rs = args.rs
np.random.seed(rs)

mat = Mat(d)
T_x0 = np.array([[d/2, 0.]])
eps = [0.02, 2.]

Sigma = (sigma**2)*np.eye(mat.D)
proposal = GaussianProposal(Sigma, mat.lb, mat.ub)

eps1 = [2., d/2]
epsT = [0.02, 2.]
all_eps = np.stack([np.linspace(eps1[i], epsT[i], T) for i in range(len(eps1))], axis=1)

N = 50
time0 = time.time()
zs = ABC_SMC(N, mat, proposal, T_x0, all_eps)

fname = "SMCABC_mat_d=%d_T=%d_sigma=%.2e_rs=%d.npz" % (d, T, sigma, rs)
if zs is not None:
    time_per_samp = (time.time() - time0)/N
    print(zs.shape)
    xs = mat.simulate(zs[-1])
    np.savez(fname, zs=zs, xs=xs, time_per_samp=time_per_samp)
else:
    np.savez(fname, zs=0, xs=0, time_per_samp=np.nan)
