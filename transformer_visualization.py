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
from itertools import compress, product

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


def get_diversity(data, bin_step, attn_max=None, attn_min=None, scale='log', model_name=''):
    """
    get the diversity based on the attention distribution
    """
    offset = 1e-8
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

    # extract spread index
    spread_idx = np.apply_along_axis(lambda x: np.argmax(x >= 0.5), -1, np.cumsum(attn_hists, axis=-1))
 
    with open("sparsity_spread/"+model_name.replace('/', '-')+".npy", "wb+") as f:
        np.save(f, spread_idx, allow_pickle=False)

    spread_idx = np.std(spread_idx, axis = -1)
    spread_idx = np.mean(spread_idx, axis = -1)
    return spread_idx


def get_token_sparse_count_percentage(data):
    all_token_count, all_token_percentage = [], []
    for inst in data:
        curr_token_count = np.cumsum(np.flip(np.sort(inst, axis=-1), axis=-1), axis=-1)
        curr_token_count = np.apply_along_axis(lambda x: np.argmax(x > 0.5), -1, curr_token_count)
        print(curr_token_count.shape)
        curr_token_percentage = curr_token_count / inst.shape[-1]
        print(curr_token_percentage.shape)
        all_token_count.append(curr_token_count)
        all_token_percentage.append(curr_token_percentage)
    
    all_token_count = np.concatenate(all_token_count, axis=-1)
    all_token_percentage = np.concatenate(all_token_percentage, axis=-1)
    return all_token_count, all_token_percentage


def get_focused_token_mean_std(count, percentage, model_name=''):
    model_name = model_name.split('/')[-1]
    mean_count, mean_percentage = np.mean(count, axis=-1), np.mean(percentage, axis=-1)
    std_count, std_percentage = np.std(count, axis=-1), np.std(percentage, axis=-1)
    fig, ax = plt.subplots(1, 1, figsize=(24, 4))
    indices = ["L{}H{}".format(i, j) for i, j in list(product(np.arange(1, 13), range(1, 13)))]
    indices = ["{}".format(i+1) for i in range(144)]
    for i in range(144):
        if i%12 == 6: indices[i] = "layer {}".format(int(i/12) + 1)

    # ax.errorbar(indices, mean_count.flatten(), fmt='.', ecolor='grey', capsize=3, lw=1)
    ax.errorbar(indices, mean_count.flatten(), yerr=std_count.flatten(), fmt='ok', lw=3)
    ax.grid(linestyle='--', color='grey', alpha=0.4)
    ax.margins(0.002)
    # ax.set_yticks([0] + [2**i for i in np.arange(1, 8, 1)])
    # ax.set_yscale('log', basey=2)
    # ax.set_ylim([1, 90])
    ax.set_ylabel('average #tokens \nto majority', fontsize=22)
    for l in range(12):
        ax.axvspan(l*12-0.5, l*12+12-0.5, alpha=0.2, facecolor='C{}'.format(l))
    ax.set_xticklabels(indices, Fontsize=22)
    for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        if idx % 12 !=6:
            tick.label1.set_visible(False)
        else: tick.label1.set_visible(True)
        
    for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(22)

    fig.tight_layout()
    fig.savefig(RES_FIG_PATH + 'head_consistency_count_{}.pdf'.format(model_name))
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(24, 4))
    # ax.errorbar(indices, mean_count.flatten(), fmt='.', ecolor='grey', capsize=3, lw=1)
    ax.errorbar(indices, mean_percentage.flatten(), yerr=std_percentage.flatten(), fmt='ok', lw=3)
    ax.grid(linestyle='--', color='grey', alpha=0.4)
    ax.margins(0.002)
    ax.set_ylabel('average proportion of \ntokens to majority', fontsize=21)
    # ax.set_ylim((0, 100))
    for l in range(12):
        ax.axvspan(l*12-0.5, l*12+12-0.5, alpha=0.2, facecolor='C{}'.format(l))
    ax.set_xticklabels(indices, Fontsize=22)
    for idx, tick in enumerate(ax.xaxis.get_major_ticks()):
        if idx % 12 !=6:
            tick.label1.set_visible(False)
        else: tick.label1.set_visible(True)
        
    for idx, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label.set_fontsize(22)
        
    fig.tight_layout()
    fig.savefig(RES_FIG_PATH + 'head_consistency_percent_{}.pdf'.format(model_name))
    plt.clf()

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
    offset = 1e-8
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
    
    for layer_idx, layer in enumerate(attn_hists):
        print("plotting layer {}...".format(layer_idx))
        for head_idx, head in enumerate(layer):
            fig = plt.figure()
            gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[8.5, 1.5], figure=fig)
            curr_ax = fig.add_subplot(gs[0, 0])
            curr_ax2 = curr_ax.twinx()
            alpha_val = 0.01
            for row in head:
                curr_ax.plot(attn_bins[:-1], row, atten_bar_width,
                             color='C0', linewidth=0.5, linestyle='-', alpha=alpha_val)
                curr_ax2.plot(attn_bins[:-1], np.cumsum(row),
                             color='C3', linewidth=0.5, linestyle='-', alpha=alpha_val)

            curr_ax.tick_params(labelsize=16)
            curr_ax2.tick_params(labelsize=16)

            # plot sparse hist if exist:
            if sparse_hist is not None:
                sparse_hist_ax = fig.add_subplot(gs[0, 1])
                for i, sparsity in enumerate(sparse_hist[layer_idx][head_idx]):
                    sparse_hist_ax.bar(1, width=2, height=0.1, bottom=0.1*i, alpha=sparsity, color='r')
                    sparse_hist_ax.bar(1, width=2, height=0.1, bottom=0.1*i, fill=False, color='r')
                    sparse_hist_ax.text(0.1, 0.1*(i+1)-0.07, '{:.2f}'.format(sparsity), fontsize=15)

                sparse_hist_ax.tick_params(labelsize=14)
                sparse_hist_ax.yaxis.tick_right()
                sparse_hist_ax.set_xlim([0, 1])
                sparse_hist_ax.get_xaxis().set_visible(False)
                sparse_hist_ax.set_ylim([0, 1.0])
                sparse_hist_ax.set_title('sparsity\ndistribution', fontsize=15)

            # subplot_title = 'max: {:.4f}, min: {:.4f}'.format(
            #     attn_max[layer_idx][head_idx], attn_min[layer_idx][head_idx], fontsize=10)

            # curr_ax.set_title('\n'.join(wrap(subplot_title, 38)))
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

            # fig.suptitle("Histogram for layer {} head {}(per token){}".format(
                # layer_idx, head_idx, attached_title), fontsize=15, y=0.97)
            fig.tight_layout(pad=2.2)
            plt.savefig(
                RES_FIG_PATH+'at_hist_per_token_layer_{}_head_{}.png'.format(layer_idx, head_idx), dpi=600)
            plt.clf()
            plt.close(fig)


def plot_atten_dist_per_token_compare_heads(data, bin_step, heads_idx, attn_max=None, attn_min=None, scale='log', attached_title='', ylim=0.2):
    """
    plotting the attention histogram per token, stacking all plots together and compare 
    multiple heads in a same plot.
    accepted data: a list of attention matrices, with each as [layer, head, length, length]
    or a numpy array storing rows of histograms in [layer, head, length, bin_step]
    attn_max and attn_min provides the max/min values per head across all insts. 
    They both are in [layer, head]
    attn_max and attn_min are not required when data is a list of matrice
    """
    offset = 1e-8
    hist_x_start, hist_x_end = log(offset, 10), log(1, 10)
    if scale == 'linear':
        offset = 0.0

    attn_bins, attn_hists = get_bin_edges(bin_step, hist_x_start, hist_x_end, scale), None

    if type(data) is list:
        for inst in data:
            inst_attn_hist = np.apply_along_axis(
                lambda x: np.histogram(x + offset, attn_bins, range=(0.0, 1.0))[0], -1,  inst)

            attn_hists = inst_attn_hist if attn_hists is None else \
                np.concatenate([attn_hists, inst_attn_hist], axis=-2)

        # Normalization
        attn_hists = np.apply_along_axis(lambda a: a / np.sum(a), -1, attn_hists)
    else:
        attn_hists = data

    print(attn_hists.shape)
    atten_bar_width = [attn_bins[i] - attn_bins[i-1] for i in range(1, len(attn_bins))]

    heads_to_print = [attn_hists[l, h, :, :] for l, h in heads_idx]

    fig = plt.figure()
    curr_ax = fig.add_subplot(111)
    alpha_val = 0.01
    for head_idx, head in enumerate(heads_to_print):
        for row in head:
            curr_ax.plot(attn_bins[:-1], row, atten_bar_width,
                             color='C{}'.format(head_idx), linewidth=0.5, linestyle='-', alpha=alpha_val)

    patches = [mpatches.Patch(color='C{}'.format(idx), label='layer {} head {}'.format(l, h)) for idx, (l, h) in enumerate(heads_idx)]

    curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
    curr_ax.set_xscale(scale)
    # curr_ax.set_yscale('log')
    curr_ax.set_yticks(np.linspace(0, ylim, 11))
    curr_ax.set_ylim((0, ylim))

    if scale == 'log':
        curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1), 1])
    else:
        curr_ax.set_xlim([0, 0.02])

    fig.legend(handles=patches, loc='upper left', ncol=1, bbox_to_anchor=(0.11, 0.93))
    fig.tight_layout(pad=2.2)
    plt.savefig(
        RES_FIG_PATH+'at_hist_per_token_compare.png', dpi=600)
    plt.clf()
    plt.close(fig)


def plot_atten_dist_per_token_compare_models(data_att0, data_att1, bin_step, scale='log', attached_title='', ylim=0.3):
    """
    plotting the attention histogram per token, stacking all plots together and compare attentions for different models
    accepted data: a list of attention matrices, with each as [layer, head, length, length]
    """
    offset = 1e-10
    hist_x_start, hist_x_end = log(offset, 10), log(1, 10)
    if scale == 'linear':
        offset = 0.0

    attn_bins, attn_hists = get_bin_edges(bin_step, hist_x_start, hist_x_end, scale), []

    for data in [data_att0, data_att1]:
        single_attn_hist = None
        for inst in data:
            inst_attn_hist = np.apply_along_axis(
                lambda x: np.histogram(x + offset, attn_bins, range=(0.0, 1.0))[0], -1,  inst)

            single_attn_hist = inst_attn_hist if single_attn_hist is None else \
                np.concatenate([single_attn_hist, inst_attn_hist], axis=-2)

        # Normalization
        attn_hists.append(np.apply_along_axis(lambda a: a / np.sum(a), -1, single_attn_hist))

    print(attn_hists[0].shape)
    atten_bar_width = [attn_bins[i] - attn_bins[i-1] for i in range(1, len(attn_bins))]
    
    alpha_val = 0.01
    
    for layer_idx, (layer_att0, layer_att1) in enumerate(zip(*attn_hists)):
        for head_idx, (head_att0, head_att1) in enumerate(zip(layer_att0, layer_att1)):
            fig = plt.figure()
            curr_ax = fig.add_subplot(111)
            for row in head_att0:
                curr_ax.step(attn_bins[:-1], row, atten_bar_width,
                                color='C0', linewidth=0.5, linestyle='-', alpha=alpha_val)
            for row in head_att1:
                curr_ax.step(attn_bins[:-1], row, atten_bar_width,
                                color='C1', linewidth=0.5, linestyle='-', alpha=alpha_val)

            patches = [mpatches.Patch(color='C0', label='original'), mpatches.Patch(color='C1', label='quantized')]

            curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
            curr_ax.set_xscale(scale)
            # curr_ax.set_yscale('log')
            curr_ax.set_yticks(np.linspace(0, ylim, 11))
            curr_ax.set_ylim((0, ylim))

            if scale == 'log':
                curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1), 1])
            else:
                curr_ax.set_xlim([0, 0.02])

            curr_ax.legend(handles=patches, loc='upper left', ncol=1)
            fig.tight_layout()
            plt.savefig(RES_FIG_PATH + \
                        'at_hist_per_token_compmodel_layer{}head{}{}.png'.format(layer_idx, head_idx, attached_title), \
                        dpi=600)
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

def plot_atten_dist_per_token_with_names(atten_hists, token_names, offset, head_idx=(0, 0)):
    """
    plot attention distribution with token names
    """
    if atten_hists.shape[0] < len(token_names):
        raise ValueError("token length in the attention should be larger than token list length!")

    fig = plt.figure()
    curr_ax = fig.add_subplot(111)
    # curr_ax2 = curr_ax.twinx()
    hist_x_start, hist_x_end = log(offset, 10), log(1, 10)
    attn_bins = get_bin_edges(100, hist_x_start, hist_x_end, 'log')
    atten_bar_width = [attn_bins[i] - attn_bins[i-1] for i in range(1, len(attn_bins))]

    special_token_list, insert_counter = {}, 0
    for token in token_names: 
        if not token in special_token_list:
            special_token_list[token] = 'C{}'.format(insert_counter)
            insert_counter += 1
    
    for token_idx, row in enumerate(atten_hists):
        if token_idx < len(token_names):
            curr_ax.plot(attn_bins[:-1], np.cumsum(row), atten_bar_width,
                        color=special_token_list[token_names[token_idx]], linewidth=0.5, linestyle='-', alpha=0.8)
        else:
            curr_ax.plot(attn_bins[:-1], np.cumsum(row), atten_bar_width,
                        color='black', linewidth=0.5, linestyle='-.', alpha=0.3)
        # curr_ax2.plot(attn_bins[:-1], np.cumsum(row),
        #              color='C3', linewidth=0.5, linestyle='-', alpha=alpha_val)

    curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
    curr_ax.set_yticks(np.linspace(0, 1.0, 11))
    # curr_ax2.set_yticks(np.linspace(0, ylim[1], 11))
    curr_ax.set_ylim((0, 1.0))
    # curr_ax2.set_ylim((0, ylim[1]))
    curr_ax.set_xscale('log')
    curr_ax.set_xlim([10 ** hist_x_start - 10 ** (hist_x_start-1), 1])

    patches = [mpatches.Patch(color=c, label=token) for token, c in special_token_list.items()]
    curr_ax.legend(handles=patches, loc='upper left', ncol=1, prop={'size': 7})
    fig.suptitle("Histogram for layer {} head {}(per token)".format(head_idx[0], head_idx[1]))
    plt.savefig(
        RES_FIG_PATH+'at_hist_named_token_layer_{}_head_{}.png'.format(head_idx[0], head_idx[1]), dpi=600)
    plt.clf()
    plt.close(fig)


def plot_dist_diversity(data: dict, attached_title=''):
    """
    """
    curr_ax = plt.subplot(111)
    fig = plt.gcf()
    plt.xticks(fontsize=17)
    
    patches = [mpatches.Patch(color='C{}'.format(i), label=model) for i, model in enumerate(data.keys())]
 
    for model_idx, model in enumerate(data.keys()):
        curr_ax.plot(np.arange(1, 13), data[model], color='C{}'.format(model_idx), marker='s', markersize=7)
    for label in curr_ax.yaxis.get_majorticklabels(): label.set_fontsize(17)
    
    # curr_ax.set_ylim([0, 5])
    curr_ax.set_xticks(np.arange(1, 13))
    # curr_ax.set_title('head_{}'.format(head_idx))
    curr_ax.set_xlabel('layer', fontsize=17)
    curr_ax.set_ylabel('average diversity \nof all heads', fontsize=17)
    curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
    curr_ax.legend(handles=patches, loc='upper left', ncol=1, fontsize=17)
    
    fig.tight_layout()
    plt.savefig(RES_FIG_PATH+'dist_spread.pdf')
    plt.clf()
    plt.close(fig)


def plot_spread_features(stat_features, model_name):
    fig, ax = plt.subplots(1, 1, figsize=(24, 4))
    means, stds, maxs, mins = (i.flatten() for i in stat_features)
    indices = ["L{}H{}".format(i, j) for i, j in list(product(np.arange(1, 13), range(1, 13)))]
    ax.errorbar(indices, means, yerr=[means - mins, maxs - means],
                   fmt='.', ecolor='grey', capsize=3, lw=1)
    ax.errorbar(indices, means, yerr=stds, fmt='ok', lw=3)
    ax.grid(linestyle='--', color='grey', alpha=0.4)
    ax.margins(0.002)
    ax.set_ylim((0, 100))
    for l in range(12):
        ax.axvspan(l*12-0.5, l*12+12-0.5, alpha=0.2, facecolor='C{}'.format(l))
    ax.set_xticklabels(indices, rotation=60)
    ax.set_yticks(np.arange(0, 110, 10))
    ax.set_yticklabels(['1e{}'.format(i) for i in np.arange(-20, 1, 2)])

    fig.tight_layout()
    fig.savefig(RES_FIG_PATH + 'head_consistency_stat_{}.pdf'.format(model_name))
    plt.clf()
    
    fig, ax = plt.subplots(1, 1, figsize=(24, 4))
    ax.hist(means, 100, weights=np.ones(means.shape)*1/144.0, range=(0, 100), color='black')
    ax.grid(linestyle='--', color='grey', alpha=0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 110, 10))
    ax.set_xticklabels(['1e{}'.format(i) for i in np.arange(-20, 1, 2)])
    fig.tight_layout()
    fig.savefig(RES_FIG_PATH + 'head_consistency_dist_{}.pdf'.format(model_name))
    plt.clf()


def plot_em_sparsity(sparsity_data: dict, second_axis_data={}, attached_title='', normalize_score=False, append_to_fname='', **kwargs):
    # plot em vs. sparsity
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.xticks(fontsize=15)
    patches = []
    ax.set_xlabel("sparsity", fontsize=15)
    
    for idx, (data_label, data) in enumerate(sparsity_data.items()):
        ax.set_ylabel("EM score", fontsize=15)
        patches.append(mpatches.Patch(color='C{}'.format(idx), label=data_label))
        scores = data['em']/data['em'].max() if normalize_score else data['em'] * 100
        ax.plot(data['all'], scores,
                color='C{}'.format(idx), marker='s', markersize=4)

    for label in ax.yaxis.get_majorticklabels(): label.set_fontsize(15)
    
    if len(second_axis_data.keys()) > 0:
        ax2 = ax.twinx()
        ax2.set_ylabel('pseudo-perplexity', fontsize=15)
        for idx2, (data_label, data) in enumerate(second_axis_data.items()):
            patches.append(mpatches.Patch(color='C{}'.format(idx+idx2+1), label=data_label))
            scores = data['em']/data['em'].max() if normalize_score else data['em']
            ax2.plot(data['all'], scores,
                    color='C{}'.format(idx+idx2+1), marker='s', markersize=4)
    
    for label in ax2.yaxis.get_majorticklabels(): label.set_fontsize(15)
    ax2.invert_yaxis()


    # ax.set_ylim([30, 90])
    # fig.suptitle(
    #     'Accuracy vs. Sparsity {}'.format(attached_title))
    fig.tight_layout()
    plt.legend(handles=patches, loc='lower left', **kwargs)
    plt.grid(linestyle='--', alpha=0.5, color='grey')
    plt.savefig(RES_FIG_PATH+'performance_vs_sparsity{}.pdf'.format(append_to_fname))
    plt.close(fig)


def plot_em_quant(sparsity_data: dict, attached_title='', normalize_score=False, append_to_fname='', **kwargs):
    # plot em vs. quant
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.xticks(fontsize=15)
    patches = []
    ax.set_xlabel("#bits", fontsize=15)
    ax.invert_xaxis()
    
    for idx, (data_label, data) in enumerate(sparsity_data.items()):
        quant_bits = [int(i) for i in data.index]
        ax.set_ylabel("EM score", fontsize=15)
        patches.append(mpatches.Patch(color='C{}'.format(idx), label=data_label))
        scores = data['em']/data['em'].max() if normalize_score else data['em'] * 100
        ax.plot(quant_bits, scores,
                color='C{}'.format(idx), marker='s', markersize=4)

    for label in ax.yaxis.get_majorticklabels(): label.set_fontsize(15)

    # ax.set_ylim([30, 90])
    # fig.suptitle(
    #     'Accuracy vs. Sparsity {}'.format(attached_title))
    fig.tight_layout()
    plt.legend(handles=patches, loc='lower left', **kwargs)
    plt.grid(linestyle='--', alpha=0.5, color='grey')
    plt.savefig(RES_FIG_PATH+'performance_vs_quantbits{}.pdf'.format(append_to_fname))
    plt.close(fig)


def quantize_attention(atts: list):
    def uniform_quant(att, base):
        ret_att = np.floor(att / base + 0.5) * base
        return ret_att

    def log_quant(att, bits):
        exp = np.floor(np.log2(att) + 0.5)
        min_exp = -(2.0**bits-1)
        clamped_exp = np.copy(exp)
        clamped_exp[exp < min_exp] = min_exp
        return np.power(2.0, clamped_exp)

    bits = 16
    base = 1.0/(2**bits)
    # ret = [uniform_quant(att, base) for att in atts]
    ret = [log_quant(att, bits) for att in atts]
    return ret


