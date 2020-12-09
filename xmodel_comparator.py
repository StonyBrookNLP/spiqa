"""
python script used to compare the performance across different models
"""

import numpy as np
import pandas as pd
import transformer_visualization as tv

import os

def get_em_sparsities(params_path: str, sparsity_bar=0.025, layer_aggregration='mean', avg_score=False):
    '''
    extract sparsities for a fixed sparsity bar from all parameters with different threshold.
    '''
    params_path_list = os.listdir(params_path)
    threshold_list = [i.replace('_', '.') for i in params_path_list]
    sparsity_table = pd.DataFrame(index=[i for i in threshold_list])
    params_path_list = [params_path + '/' + i + '/' for i in params_path_list]

    for threshold, params in zip(threshold_list, params_path_list):
        # read from file
        input_type = "_all"
        score_path = (params + 'score' + input_type + '.npy')
        att_stat_path = (params + 'att_stat_features' + input_type + '.npy')
        if os.path.isfile(att_stat_path) and os.path.isfile(score_path):
            with open(score_path, "rb") as score_file:
                total_score, qa_pair_count = (i for i in np.load(score_file))
            with open(att_stat_path, "rb") as att_stat_file:
                all_max = np.load(att_stat_file)
                all_min = np.load(att_stat_file)
                all_mean = np.load(att_stat_file)
                all_std = np.load(att_stat_file)
                all_sparsity = np.load(att_stat_file)

        for layer_idx, layer in enumerate(all_sparsity):
            for head_idx, spars_per_head in enumerate(layer):
                sparsity_table.at[threshold, 'layer_{}_head_{}'.format(
                    layer_idx, head_idx)] = spars_per_head

        sparsity_table.at[threshold, 'all'] = np.mean(all_sparsity.flatten())
        sparsity_table.at[threshold, 'rmheads'] = np.sum(all_sparsity.flatten()) / 4.0
        sparsity_table.at[threshold, 'em'] = total_score / qa_pair_count if avg_score else total_score

    return sparsity_table

def get_em_quantbits(params_path: str, layer_aggregration='mean', avg_score=False):
    '''
    extract em scores from all parameters for different quant bits
    '''
    params_path_list = os.listdir(params_path)
    threshold_list = [int(i.replace('_', '.')) for i in params_path_list]
    sparsity_table = pd.DataFrame()
    params_path_list = [params_path + '/' + i + '/' for i in params_path_list]

    for threshold, params in zip(threshold_list, params_path_list):
        # read from file
        input_type = "_all"
        score_path = (params + 'score' + input_type + '.npy')
        att_stat_path = (params + 'att_stat_features' + input_type + '.npy')
        if os.path.isfile(att_stat_path) and os.path.isfile(score_path):
            with open(score_path, "rb") as score_file:
                total_score, qa_pair_count = (i for i in np.load(score_file))
        sparsity_table.at[threshold, 'em'] = total_score / qa_pair_count if avg_score else total_score

    return sparsity_table.dropna().sort_index(ascending=False)

if __name__ == '__main__':
    roberta_quant = get_em_quantbits('./quantized_params/roberta-base-squad', avg_score=True)
    print(roberta_quant)
    tv.plot_em_quant({'RoBERTa-SQuAD': roberta_quant}, fontsize=15)