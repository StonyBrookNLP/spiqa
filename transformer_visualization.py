'''
Plotting function for transformer hstate and attention visualization
'''
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from textwrap import wrap
from math import isnan, fsum, log, ceil, floor


RES_FIG_PATH = "./res_fig/"


def get_bin_edges(bin_step, hist_x_start, hist_x_end, scale):
    if type(bin_step) is int:
        if scale == 'log':
            bin_edges = 10**np.linspace(hist_x_start, hist_x_end, bin_step+1)
            bin_edges[0] -= 10**(hist_x_start-1)
            return bin_edges
        else:
            return np.linspace(hist_x_start, hist_x_end, bin_step+1)
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


def plot_heatmap(data, sparsity_bar=0.025, auto_scale=False, binarize=True, layer_aggregration='mean', attached_title=''):
    '''
    Plot the heat map to visualize the relation between each subwords in the
    self attention of each attention head in each layer

    expected data shape: (#layers, #heads, length, length)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    auto_scale: whether to auto scale the color bar
    binarize: if true, all values > sparsity_bar will be 1 and < will be 0
    '''
    for layer_idx, layer in enumerate(data):
        fig, axs = plt.subplots(3, 4, figsize=(19, 12))
        print("Plotting heatmap for layer {}...".format(layer_idx))
        for head_idx, head in enumerate(layer[0]):
            sparsity = (head <= sparsity_bar).sum() / head.flatten().shape[0]
            info = 'head_{}, max: {:.4f}, min: {:.4f}, spars: {:.4f}, sparsity_bar: {:.4f}'.format(
                head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar)
            if binarize:
                head = np.array((head > sparsity_bar)).astype("float")
            ax = axs[int(head_idx/4), int(head_idx % 4)]
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            c = ax.pcolormesh(head) if auto_scale else ax.pcolormesh(
                head, vmin=0.0, vmax=1.0)
            fig.colorbar(c, ax=ax)
            ax.set_title('\n'.join(wrap(info, 35)))

        fig.suptitle('Heatmap of Layer {}\'s Attention per head (batch aggregation={}, {})'
                     .format(layer_idx, layer_aggregration, attached_title), fontsize=21, y=0.99)
        fig.tight_layout()
        fig_path = RES_FIG_PATH+"auto_scale_" if auto_scale else RES_FIG_PATH
        fig_path = fig_path+"bin_" if binarize else fig_path
        plt.savefig(fig_path+'heatmap_layer{}.png'.format(layer_idx), dpi=600)
        plt.clf()
        plt.close(fig)


def plot_atten_dist_per_token(data, bin_step, attn_max=None, attn_min=None, sparse_hist=None, scale='log', attached_title='', model_name='', ylim=(0.2, 1)):
    """
    plotting the attention histogram per token, stacking all plots together.
    accepted data: a list of attention matrices, with each as [layer, head, length, length]
    or a numpy array storing rows of histograms in [layer, head, length, bin_step]
    attn_max and attn_min provides the max/min values per head across all insts. 
    They both are in [layer, head]
    attn_max and attn_min are not required when data is a list of matrice
    """
    offset = 1e-10
    hist_x_start, hist_x_end = log(offset, 10), log(1, 10)
    if scale == 'linear':
        offset = 0.0

    attn_bins, attn_hists = get_bin_edges(bin_step, hist_x_start, hist_x_end, scale), None

    if type(data) is list:
        for inst in data:
            inst_attn_hist = np.apply_along_axis(
                lambda x: np.histogram(x + offset, attn_bins, range=(0.0, 1.0))[0], -1,  inst)
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

    # extract spread index
    spread_idx = np.apply_along_axis(lambda x: np.argmax(x > 0.5), -1, np.cumsum(attn_hists, axis=-1))
    spread_idx = np.std(spread_idx, axis = -1)
    
    with open("sparsity_spread/"+model_name.replace('/', '-')+".txt", "w+", newline='') as f:
        spread_idx = pd.DataFrame(spread_idx, columns=['head_{}'.format(i) for i in range(spread_idx.shape[-1])])
        f.write(spread_idx.to_string())

    return
    
    for layer_idx, layer in enumerate(attn_hists):
        print("plotting layer {}...".format(layer_idx))
        for head_idx, head in enumerate(layer):
            fig = plt.figure(constrained_layout=True)
            gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[9, 1], figure=fig)
            curr_ax = fig.add_subplot(gs[0, 0])
            curr_ax2 = curr_ax.twinx()
            alpha_val = 0.01
            for row in head:
                curr_ax.plot(attn_bins[:-1], row, atten_bar_width,
                             color='C0', linewidth=0.5, linestyle='-', alpha=alpha_val)
                curr_ax2.plot(attn_bins[:-1], np.cumsum(row),
                             color='C3', linewidth=0.5, linestyle='-', alpha=alpha_val)

            # plot sparse hist if exist:
            if sparse_hist is not None:
                sparse_hist_ax = fig.add_subplot(gs[0, 1])
                for i, sparsity in enumerate(sparse_hist[layer_idx][head_idx]):
                    sparse_hist_ax.bar(0.5, width=1, height=0.1, bottom=0.1*i, alpha=sparsity, color='r')
                    sparse_hist_ax.bar(0.5, width=1, height=0.1, bottom=0.1*i, fill=False, color='r')
                    sparse_hist_ax.text(0.2, 0.1*(i+1)-0.07, '{:.2f}'.format(sparsity))

                sparse_hist_ax.yaxis.tick_right()
                sparse_hist_ax.set_xlim([0, 1])
                sparse_hist_ax.get_xaxis().set_visible(False)
                sparse_hist_ax.set_ylim([0, 1.0])
                sparse_hist_ax.set_title('sparsity\ndistribution', fontsize=10)

            subplot_title = 'max: {:.4f}, min: {:.4f}'.format(
                attn_max[layer_idx][head_idx], attn_min[layer_idx][head_idx], fontsize=10)

            curr_ax.set_title('\n'.join(wrap(subplot_title, 38)))
            curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
            curr_ax.set_xscale(scale)
            # curr_ax.set_yscale('log')
            curr_ax.set_yticks(np.linspace(0, ylim[0], 11))
            curr_ax2.set_yticks(np.linspace(0, ylim[1], 11))
            curr_ax.set_ylim((0, ylim[0]))
            curr_ax2.set_ylim((0, ylim[1]))

            if scale == 'log':
                curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1), 1])
            else:
                curr_ax.set_xlim([0, 0.02])

            fig.suptitle("Histogram for layer {} head {}(per token){}".format(
                layer_idx, head_idx, attached_title), fontsize=15, y=0.97)
            fig.tight_layout()
            plt.savefig(
                RES_FIG_PATH+'at_hist_per_token_layer_{}_head_{}.png'.format(layer_idx, head_idx), dpi=600)
            plt.clf()
            plt.close(fig)


def plot_hs_dist_per_token(data, bin_step, attn_mask, scale='log', attached_title='', ylim=(0.2, 1)):
    """
    plotting hidden states per token's distribution.
    data: [#layer+1, batch_size, length, hidden_state_size]
    attn_mask: a list of available length for each results for 1 batch. len(attn_mask) == batch_size
    """
    if len(attn_mask) != data.shape[1]:
        raise ValueError(
            "Error: the attention mask should have same length as the batch size in the hidden states.")

    hs_max, hs_min = np.amax(data[1:, :, :, :], axis=(-3, -2, -1)), np.amin(data[1:, :, :, :], axis=(-3, -2, -1))
    # hist_x_start, hist_x_end = float(floor(np.amin(data))), float(ceil(np.amax(data)))
    hist_x_start, hist_x_end = -2.5, 2.5
    hs_bins, hs_hists = get_bin_edges(bin_step, hist_x_start, hist_x_end, scale), None
    hs_hists = np.apply_along_axis(
                lambda x: np.histogram(x, bins=hs_bins, weights=(np.ones_like(x) / len(x)))[0], -1, data)

    hs_bar_width = [hs_bins[i] - hs_bins[i-1] for i in range(1, len(hs_bins))]

    fig, ax = plt.subplots(3, 4, figsize=(22, 15))
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams.update({'xtick.labelsize': 13})
    matplotlib.rcParams.update({'ytick.labelsize': 13})
    for layer_idx, layer in enumerate(hs_hists[1:, :, :, :]):
        curr_ax = ax[int(layer_idx / 4), int(layer_idx % 4)]
        curr_ax2 = curr_ax.twinx()
        alpha_val = 0.01
        for mask, inst in zip(attn_mask, layer):
            for row_idx, row in enumerate(inst):
                color_id = 0 if row_idx < mask else 5
                curr_ax.plot(hs_bins[:-1], row, 
                        color='C{}'.format(color_id), linewidth=0.5, linestyle='-', alpha=alpha_val)
                curr_ax2.plot(hs_bins[:-1], np.cumsum(row),
                        color='C{}'.format(color_id+3), linewidth=0.5, linestyle='-', alpha=alpha_val)

        subplot_title = 'layer_{}, max: {:.4f}, min: {:.4f}, \n#elem<left: {:.4f}, #elem>right: {:.4f}'.format(
            layer_idx, hs_max[layer_idx], hs_min[layer_idx], \
            np.count_nonzero(data[layer_idx+1] < hist_x_start) / float(10*320*768), \
            np.count_nonzero(data[layer_idx+1] > hist_x_end) / float(10*320*768))
        curr_ax.set_title('\n'.join(wrap(subplot_title, 42)))
        curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
        curr_ax.set_xscale(scale)
        # curr_ax.set_yscale('log')
        curr_ax.set_yticks(np.linspace(0, ylim[0], 11))
        curr_ax2.set_yticks(np.linspace(0, ylim[1], 11))
        curr_ax.set_ylim((0, ylim[0]))
        curr_ax2.set_ylim((0, ylim[1]))
        if scale == 'log':
            curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1),
                              10 ** hist_x_end])
        else:
            curr_ax.set_xlim([hist_x_start, hist_x_end])
    
    fig.suptitle("Histogram for Hidden States per layer (per token){}".format(attached_title), fontsize=21, y=0.99)
    patches = [mpatches.Patch(color='C0', label='pdf of tokens within actual size'), 
                mpatches.Patch(color='C3', label='cdf of tokens within actual size'), 
                mpatches.Patch(color='C5', label='pdf of padded tokens'),
                mpatches.Patch(color='C8', label='cdf of padded tokens')]
    fig.legend(handles=patches, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(pad=2.2)
    plt.savefig(RES_FIG_PATH+'hs_hist_per_token.png', dpi=600)
    plt.clf()
    plt.close(fig)
