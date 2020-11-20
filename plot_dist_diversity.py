import transformer_visualization as tv
import os
import numpy as np

params_path = './sparsity_spread/cdfs-same-range'
model_list = [
                params_path + '/roberta-base.npy', 
                params_path + '/roberta-base-squad-v1.npy', 
                params_path + '/roberta-base-SST-2.npy',
                params_path + '/bert-base-uncased.npy',
                params_path + '/csarron-bert-base-uncased-squad-v1.npy'
                ]
patches = [
            'roberta', 
            'roberta-squad', 
            'roberta-sst2', 
            'bert', 
            'bert-squad'
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

