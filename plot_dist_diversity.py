import transformer_visualization as tv
import os
import numpy as np

params_path = 'sparsity_spread'
model_list = os.listdir(params_path)

dist_diversity = {}

for model in model_list:
    with open(params_path + '/' + model, 'rb') as f:
        spread_idx = np.load(f)

    dist_diversity[model.split('.')[0]] = spread_idx

print(dist_diversity.keys())
tv.plot_dist_diversity(dist_diversity)