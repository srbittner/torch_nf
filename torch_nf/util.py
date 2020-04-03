import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_dist(z, log_q_z):
    df = pd.DataFrame(z)
    D = z.shape[1]
    z_labels = ["z%d" % d for d in range(1, D + 1)]
    df.columns = z_labels
    df["log_q_z"] = log_q_z

    log_q_z_std = log_q_z - np.min(log_q_z)
    log_q_z_std = log_q_z_std / np.max(log_q_z_std)
    cmap = plt.get_cmap("viridis")
    g = sns.PairGrid(df, vars=z_labels)
    g = g.map_diag(sns.kdeplot)
    g = g.map_upper(plt.scatter)

    return g
