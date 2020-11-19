import transformer_visualization as tv
import os
import numpy as np

params_path = './sparsity_spread'
model_list = [params_path + '/roberta-base.npy', params_path + '/roberta-base-squad-v1.npy', params_path + '/roberta-base-SST-2.npy']
patches = ['pretrained', 'SQuAD-v1.1', 'SST-2']

dist_diversity = {}

for patch, model in zip(patches, model_list):
    with open(model, 'rb') as f:
        spread_idx = np.load(f)

    dist_diversity[patch] = spread_idx

print(dist_diversity.keys())
tv.plot_dist_diversity(dist_diversity)