import torch
import numpy as np

import torch_nf
from torch_nf.conditional_nf import NormFlow, ConditionedNormFlow
from torch_nf.error_formatters import dbg_check
from torch_nf.systems import MF_V1_4n
from torch_nf.lfi import train_APT
import scipy.stats
from torch_nf.bijectors import ToInterval

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Ma", type=int)
parser.add_argument("--H1", type=int)
parser.add_argument("--H2", type=int, default=0)
parser.add_argument("--L", type=int)
parser.add_argument("--U", type=int)
args = parser.parse_args()

M_atom = args.Ma
H1 = args.H1
H2 = args.H2
L = args.L
U = args.U

M = 2000

system = MF_V1_4n()

x0 = np.array([[0.14446039, 0.2412575,  0.36162094,
                0.23536153, 0.19078061, 0.18227517,
                0.43703067, 0.47571289, 0.65682352,
                0.1487949,  0.20043895, 0.24761808,
                0.32097366, 0.20411271, 0.17350747,
                0.31401437, 0.2942019,  0.38232728,]])

# Opt params
D = system.D
num_iters = 5000
R = 10

# two-network arch
arch_type = "autoregressive"
if H2 != 0:
    hidden_layers = [H1, H2]
else:
    hidden_layers = [H1]

rs = 1
np.random.seed(rs)
torch.manual_seed(rs)

lb = system.lb
ub = system.ub
support_layer = ToInterval(D, lb, ub)

nf = NormFlow(
    D,
    arch_type,
    True,
    num_stages=1,
    num_layers=L,
    num_units=U,
    support_layer=support_layer,
)
cnf = ConditionedNormFlow(nf, x0.shape[1], hidden_layers, dropout=False)

cnf, losses, zs, log_probs, it_time = train_APT(
    cnf, system, x0, M=M, M_atom=M_atom, R=R, num_iters=num_iters, verbose=False
)

if H2 != 0:
    hl_str = "%d_%d" % (H1, H2)
else:
    hl_str = "%d" % H1

ext = "_M=%d_Ma=%d_H=%s_MAF_L=%d_U=%d" % (M, M_atom, hl_str, L, U)
fname = "APT_V1" + ext + ".npz"

np.savez(
    fname, x0=x0, losses=losses, zs=zs, log_probs=log_probs, it_time=it_time
)
