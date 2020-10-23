'''
Plotting function for transformer hstate and attention visualization
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from textwrap import wrap
from math import isnan, fsum, log


RES_FIG_PATH = "./res_fig/"


def get_bin_edges(bin_step, hist_x_start, hist_x_end, scale):
    if type(bin_step) is int:
        if scale == 'log':
            bin_edges = 10**np.linspace(hist_x_start, hist_x_end, bin_step+1)
            bin_edges[0] -= 10**(hist_x_start-1)
            return bin_edges
        else:
            return bin_step
    elif type(bin_step) is float:
        if scale == 'log':
            bin_edges = 10**np.append(np.arange(hist_x_start,
                                                hist_x_end, bin_step), hist_x_end)
            bin_edges[0] -= 10**(hist_x_start-1)
            return bin_edges
        else:
            return np.append(np.arange(0, 1.0, bin_step), 1.0)
    else:
        return None


def plot_atten_dist_per_token(data, bin_step, attn_max=None, attn_min=None, scale='log', attached_title='', ylim=[0, 1]):
    """
    plotting the attention histogram per token, stacking all plots together.
    accepted data: a list of attention matrices, with each as [layer, head, length, length]
    or a numpy array storing rows of histograms in [layer, head, length, bin_step]
    attn_max and attn_min provides the max/min values per head across all insts. 
    They both are in [layer, head]
    attn_max and attn_min are not required when data is a list of matrice
    """
    offset = 1e-8
    hist_x_start, hist_x_end = log(offset, 10), log(1+offset, 10)
    if scale == 'linear':
        offset = 0.0

    attn_bins, attn_hists = get_bin_edges(bin_step, hist_x_start, hist_x_end, scale), None

    if type(data) is list:
        for inst in data:
            inst_attn_hist = np.apply_along_axis(
                lambda x: np.histogram(x + offset, attn_bins)[0], -1,  inst)
            inst_attn_max, inst_attn_min = \
                np.amax(inst, axis=(-2, -1)), np.amin(inst, axis=(-2, -1))

            attn_hists = inst_attn_hist if attn_hists is None else \
                np.concatenate([attn_hists, inst_attn_hist], axis=-2)
            attn_max = inst_attn_max if attn_max is None else \
                np.maximum(attn_max, inst_attn_max)
            attn_min = inst_attn_min if attn_min is None else \
                np.minimum(attn_min, inst_attn_min)

        # Normalization 
        attn_hists = np.apply_along_axis(lambda a: a / np.sum(a), -1, attn_hists)
    else:
        attn_hists = data

    print(attn_hists.shape)
    atten_bar_width = [attn_bins[i] - attn_bins[i-1] for i in range(1, len(attn_bins))]

    for layer_idx, layer in enumerate(attn_hists):
        print("plotting layer {}...".format(layer_idx))
        fig, ax = plt.subplots(3, 4, figsize=(21, 12))
        for head_idx, head in enumerate(layer):
            curr_ax = ax[int(head_idx / 4), int(head_idx % 4)]
            alpha_val = 0.01
            for row in head:
                curr_ax.plot(attn_bins[:-1], row, atten_bar_width,
                             color='C0', linewidth=0.5, linestyle='-', alpha=alpha_val)
                curr_ax.plot(attn_bins[:-1], np.cumsum(row),
                             color='C3', linewidth=0.5, linestyle='-', alpha=alpha_val)

            subplot_title = 'head_{}, max: {:.4f}, min: {:.4f}'.format(
                head_idx, attn_max[layer_idx][head_idx], attn_min[layer_idx][head_idx])

            curr_ax.set_title('\n'.join(wrap(subplot_title, 38)))
            curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
            curr_ax.set_xscale(scale)
            # curr_ax.set_yscale('log')
            curr_ax.set_ylim(ylim)
            if scale == 'log':
                curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1),
                                  10 ** hist_x_end])
            else:
                curr_ax.set_xlim([0, 0.02])

        fig.suptitle("Histogram for layer {} per head (per token){}".format(
            layer_idx, attached_title), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(
            RES_FIG_PATH+'hist_per_token_layer_{}.png'.format(layer_idx), dpi=600)
        plt.clf()
        plt.close(fig)
