import transformer_visualization as tv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from itertools import compress, product

params_path = './sparsity_spread/cdfs-same-range'
model_list = [
                params_path + '/roberta-base.npy', 
                params_path + '/roberta-base-squad-v1.npy', 
                params_path + '/roberta-base-SST-2.npy'
                # params_path + '/bert-base-uncased.npy',
                # params_path + '/csarron-bert-base-uncased-squad-v1.npy'
                ]
patches = [
            'RoBERTa', 
            'RoBERTa-SQuAD', 
            'RoBERTa-SST2' 
            # 'BERT', 
            # 'BERT-SQuAD'
        ]

spread_analysis = {}

for patch, model in zip(patches, model_list):
    with open(model, 'rb') as f:
        spread_idx = np.load(f)

    spread_mean = np.mean(spread_idx, axis=-1)
    spread_std = np.std(spread_idx, axis=-1)
    spread_max = np.amax(spread_idx, axis=-1)
    spread_min = np.amin(spread_idx, axis=-1)

    spread_analysis[patch] = (spread_mean, spread_std, spread_max, spread_min)


# print(dist_diversity)
dist_diversity = {k: np.mean(v[1], axis=-1) for k, v in spread_analysis.items()}
tv.plot_dist_diversity(dist_diversity)

for k, v in spread_analysis.items():
    tv.plot_spread_features(v, k)


#compare head_consistency
with open('params/atten_hist_sampled_per_token.npy', "rb") as hist_file:
    atten_hist = np.load(hist_file)
    atten_bins = np.load(hist_file)
    all_seq_len = np.load(hist_file)
    all_max = np.load(hist_file)
    all_min = np.load(hist_file)
    all_sparse_count = np.load(hist_file)
    sparse_hist = np.load(hist_file)
    sparse_token_count_squad = np.load(hist_file)
    sparse_token_percentage_squad = np.load(hist_file)

if os.path.isfile("params/roberta-base_attention.npy"):
    print("loading parameters from file...")
    with open("params/roberta-base" + "_attention_mask.npy", "rb") as att_mask_file:
        attn_mask = np.load(att_mask_file)
    with open("params/roberta-base" + "_attention.npy", "rb") as att_file:
        attentions_base = [np.load(att_file) for i in range(len(attn_mask))]
    with open("params/roberta-base" + "_hists.npy", "rb") as hists_file:
        hists = np.load(hists_file)
base_count, base_percentage = tv.get_token_sparse_count_percentage(attentions_base)

mean_count_squad = np.mean(sparse_token_count_squad, axis=-1).flatten()
mean_count_base = np.mean(base_count, axis=-1).flatten()
std_count_squad = np.std(sparse_token_count_squad, axis=-1).flatten()
std_count_base = np.std(base_count, axis=-1).flatten()

fig, ax = plt.subplots(1, 1, figsize=(24, 4))
indices = ["{}".format(i+1) for i in range(144)]
for i in range(144):
    if i%12 == 6: indices[i] = "layer {}".format(int(i/12) + 1)

# ax.errorbar(indices, mean_count.flatten(), fmt='.', ecolor='grey', capsize=3, lw=1)
for index, base, sq in zip(indices, mean_count_base, mean_count_squad):
    ax.errorbar(indices, mean_count_squad, yerr=std_count_squad, fmt='ok', lw=3, ecolor='red', alpha=0.05, mec='red', mfc='red')
    ax.errorbar(indices, mean_count_base, yerr=std_count_base, fmt='ok', lw=3, ecolor='blue', alpha=0.05, mfc='blue', mec='blue')

ax.grid(linestyle='--', color='grey', alpha=0.4)
ax.margins(0.002)
ax.set_ylim([-5, 90])

patches = [mpatches.Patch(color='red', label='pretrained RoBERTa'), 
                mpatches.Patch(color='blue', label='SQuAD RoBERTa')]
# ax.set_yticks([0] + [2**i for i in np.arange(1, 8, 1)])
# ax.set_yscale('log', basey=2)

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

ax.legend(handles=patches, loc='upper right', ncol=1, fontsize=22)

fig.tight_layout()
fig.savefig('res_fig/head_consistency_compare.pdf')
plt.clf()
