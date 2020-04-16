import numpy as np
import torch
from torch_nf.conditional_nf import NormFlow, ConditionedNormFlow
from torch_nf.systems import Mat
from torch_nf.lfi import train_APT
from torch_nf.bijectors import ToInterval
import scipy.stats
import time
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=2)
parser.add_argument("--rs", type=int, default=1)

args = parser.parse_args()
d = args.d
rs = args.rs
np.random.seed(rs)

mat = Mat(d)

# Opt params
M = 2000
M_atom = 100
num_iters = 5000
R = 6

# data for posterior
x0 = np.array([[0., d/2]])

# two-network arch
arch_type = 'autoregressive'
hidden_layers = [64, 64]
support_layer = ToInterval(mat.D, mat.lb, mat.ub)

np.random.seed(rs)
torch.manual_seed(rs)

nf = NormFlow(mat.D, arch_type, True, num_stages=1, 
              num_layers=2, num_units=2*mat.D, support_layer=support_layer)
nf.count_num_params()
print("# params ", nf.D_params)
cnf = ConditionedNormFlow(nf, x0.shape[1], hidden_layers, dropout=False)

cnf, losses, zs, log_probs, it_time = train_APT(
    cnf,
    mat,
    x0,
    M=M,
    M_atom=M_atom,
    R=R,
    num_iters=num_iters,
    verbose=False
)

time0 = time.time()
cnf(torch.tensor(x0).float(), M)
time_per_sample = (time.time() - time0)/M

fname = "APT_mat_d=%d_rs=%d.npz" % (d, rs)
np.savez(
    fname, x0=x0, losses=losses, zs=zs, log_probs=log_probs, 
    it_time=it_time, time_per_sample=time_per_sample,
)

