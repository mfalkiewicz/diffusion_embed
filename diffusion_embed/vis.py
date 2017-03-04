import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(mat, cmap= sns.light_palette((240, 75, 60), input="husl", as_cmap=True)):
    #mask upper triangular part
    mask = np.zeros_like(mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig_w = 11
    f, ax = plt.subplots(figsize=(fig_w, round(fig_w*1.6)))
    sns.heatmap(mat, mask=mask, cmap = cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=fig_w/1000., cbar_kws={"shrink": .5}, ax=ax)

def plot_node(path, filename):
    f = h5py.File(filename, 'r')
    cmat = f[path+'correlation'][()]
    amat = f[path+'affinity'][()]
    plot_matrix(cmat, cmap =sns.diverging_palette(240, 10, n=9,as_cmap=True))
    plot_matrix(amat)
