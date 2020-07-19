import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_dist(z, log_q_z, kde=False, z0=None, z_labels=None, inds=None, lb=None, ub=None):
    if inds is None:
        D = z.shape[1]
        inds = np.arange(D)
    else:
        D = len(inds)
    if (lb is not None and ub is not None):
        lb = lb[inds]
        ub = ub[inds]

    df = pd.DataFrame(z[:200,inds])
    if z_labels is None:
        z_labels = ["z%d" % d for d in range(1, D + 1)]
    else:
        z_labels = [z_labels[i] for i in inds]
    df.columns = z_labels
    df["log_q_z"] = log_q_z[:200]

    log_q_z_std = log_q_z - np.min(log_q_z)
    log_q_z_std = log_q_z_std / np.max(log_q_z_std)
    cmap = plt.get_cmap("viridis")
    g = sns.PairGrid(df, vars=z_labels)
    g = g.map_diag(sns.kdeplot)
    g = g.map_upper(plt.scatter, color=cmap(log_q_z_std))
    if kde:
        g = g.map_diag(sns.kdeplot)
        g = g.map_lower(sns.kdeplot)
    if z0 is not None:
        for i in range(D):
            for j in range(i+1,D):
                g.axes[i][j].plot(z0[j], z0[i], '*r', markersize=20)
                g.axes[j][i].plot(z0[i], z0[j], '*r', markersize=20)

                if (lb is not None and ub is not None):
                    g.axes[i][j].set_xlim([lb[j], ub[j]])
                    g.axes[i][j].set_ylim([lb[i], ub[i]])

                    g.axes[j][i].set_xlim([lb[i], ub[i]])
                    g.axes[j][i].set_ylim([lb[j], ub[j]])
    return g
