import torch
import numpy as np

import torch_nf
from torch_nf.conditional_nf import NormFlow, ConditionedNormFlow
from torch_nf.error_formatters import dbg_check
from torch_nf.systems import MF_V1
from torch_nf.lfi import train_SNPE
import scipy.stats
from torch_nf.bijectors import ToInterval

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--M', type=int)
parser.add_argument('--L', type=int)
parser.add_argument('--U', type=int)
args = parser.parse_args()

M = args.M
L = args.L
U = args.U

system = MF_V1()

z0 = np.array([[1.3, -1.3, 1.3, -1.0, 0.2, 0.2, 0.2, 0.2]])

x0 = system.simulate(z0)

# Opt params
D = system.D
num_iters = 50
R = 2

# two-network arch
arch_type = 'autoregressive'
hidden_layers = [128]

rs = 1
np.random.seed(rs)
torch.manual_seed(rs)

#lb = -float('inf')*np.ones((D,)) #system.lb
#ub = float('inf')*np.ones((D,)) #system.ub
lb = system.lb
ub = system.ub
support_layer = ToInterval(D, lb, ub)

nf = NormFlow(D, arch_type, True, num_stages=1,
              num_layers=L, num_units=U, support_layer=support_layer)
cnf = ConditionedNormFlow(nf, x0.shape[1], hidden_layers, dropout=False)

losses, cnf, zs, log_probs, it_time = train_SNPE(cnf, system, x0, M=M, R=R, num_iters=num_iters, z0=z0[0], verbose=False)

fname = "snpe_v1_M=%d_MAF_L=%d_U=%d.npz" % (M, L, U)
np.savez(fname, losses=losses, zs=zs, log_probs=log_probs, it_time=it_time)
