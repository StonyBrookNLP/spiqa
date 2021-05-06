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
    params_path_list = [i for i in params_path_list if i.isdigit()]
    threshold_list = [float(i.replace('_', '.')) for i in params_path_list]
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
    #roberta_squad
    roberta_squad_original = get_em_sparsities('./filtered_params/roberta-base-squad', avg_score=True)
    roberta_squad_original_em = roberta_squad_original['em'].loc['0.0'] * 100.0
    roberta_squad_quant_linear = get_em_quantbits('./quantized_params/roberta-squad-quant-linear-midval', avg_score=True)
    roberta_squad_quant_linear_clamped = get_em_quantbits('./quantized_params/roberta-squad-quant-linear-clamped-midval', avg_score=True)
    roberta_squad_quant_clamped_log_1e3 = get_em_quantbits('./quantized_params/roberta-squad-quant-log-clamped-midval', avg_score=True)
    roberta_squad_quant_log = get_em_quantbits('./quantized_params/roberta-squad-quant-log-midval', avg_score=True)
    roberta_squad_quant_bin = get_em_quantbits('./quantized_params/roberta-squad-quant-bin', avg_score=True)
    roberta_squad_quant_rank = get_em_quantbits('./quantized_params/roberta-squad-quant-rank', avg_score=True)
    tv.plot_em_quant({'RoBERTa SQuAD rank': roberta_squad_quant_rank}, break_start=15.9, break_end=9, append_to_fname="squad_rank_only")

    #bert_squad
    bert_squad_original = get_em_sparsities('./filtered_params/bert-base-uncased-squad', avg_score=True)
    bert_squad_ori_em = bert_squad_original['em'].loc['0.0'] * 100.0
    bert_squad_quant_linear = get_em_quantbits('./quantized_params/bert-squad-quant-linear-midval', avg_score=True)
    bert_squad_quant_linear_clamped = get_em_quantbits('./quantized_params/bert-squad-quant-linear-clamped-midval', avg_score=True)
    bert_squad_quant_log = get_em_quantbits('./quantized_params/bert-squad-quant-log-midval', avg_score=True)
    bert_squad_quant_log_clamped = get_em_quantbits('./quantized_params/bert-squad-quant-log-clamped-midval', avg_score=True)
    bert_squad_quant_boolean = get_em_quantbits('./quantized_params/bert-squad-quant-bin', avg_score=True)

    # roberta_mlm
    roberta_mlm_original = get_em_sparsities('./filtered_params/roberta-base-mlm')
    roberta_mlm_original_ppl = roberta_mlm_original['em'].loc['0.0']
    roberta_mlm_quant_linear = get_em_quantbits('./quantized_params/roberta-mlm-quant-linear-midval')
    roberta_mlm_quant_linear_clamped = get_em_quantbits('./quantized_params/roberta-mlm-quant-linear-clamped-midval')
    roberta_mlm_quant_log = get_em_quantbits('./quantized_params/roberta-mlm-quant-log-midval')
    roberta_mlm_quant_log_clamped = get_em_quantbits('./quantized_params/roberta-mlm-quant-log-clamped-midval')

    # #bert_mlm
    bert_mlm_original = get_em_sparsities('./filtered_params/bert-base-mlm')
    bert_mlm_original_ppl = roberta_mlm_original['em'].loc['0.0']
    bert_mlm_quant_linear = get_em_quantbits('./quantized_params/bert-mlm-quant-linear-midval')
    bert_mlm_quant_linear_clamped = get_em_quantbits('./quantized_params/bert-mlm-quant-linear-clamped-midval')
    bert_mlm_quant_log = get_em_quantbits('./quantized_params/bert-mlm-quant-log-midval')
    bert_mlm_quant_log_clamped = get_em_quantbits('./quantized_params/bert-mlm-quant-log-clamped-midval')
    
    #sst2
    roberta_sst2_original = get_em_sparsities('./filtered_params/roberta-base-sa')
    roberta_sst2_original_em = roberta_sst2_original['em'].loc['0.0'] * 100.0
    roberta_sst2_linear = get_em_quantbits('./quantized_params/roberta-sst2-quant-linear-midval')
    roberta_sst2_linear_clamped = get_em_quantbits('./quantized_params/roberta-sst2-quant-linear-clamped-midval')
    roberta_sst2_log = get_em_quantbits('./quantized_params/roberta-sst2-quant-log-midval')
    roberta_sst2_log_clamped = get_em_quantbits('./quantized_params/roberta-sst2-quant-log-clamped-midval')
    roberta_sst2_uniform_slog_clamped_mean = get_em_quantbits('./quantized_params/roberta-sst2-quant-uniform-slog-clamped-mean')
    roberta_sst2_uniform_slog_mean = get_em_quantbits('./quantized_params/roberta-sst2-quant-uniform-slog-mean')
    roberta_sst2_1bit = get_em_quantbits('./quantized_params/roberta-sst2-quant-bin')

    #hstate quantization
    # roberta_squad_hquant_linear = get_em_quantbits('./quantized_params/hidden_states/roberta-squad-hquant-linear', avg_score=True)
    # roberta_squad_hquant_evenlog = get_em_quantbits('./quantized_params/hidden_states/roberta-squad-hquant-evenlog', avg_score=True)
    # roberta_squad_hquant_evenlog_smax = get_em_quantbits('./quantized_params/hidden_states/roberta-squad-hquant-evenlog-smax', avg_score=True)
    # roberta_squad_hquant_fixed5 = get_em_quantbits('./quantized_params/hidden_states/roberta-squad-hquant-fixed5', avg_score=True)
    # roberta_squad_hquant_fixed4 = get_em_quantbits('./quantized_params/hidden_states/roberta-squad-hquant-fixed4', avg_score=True)

    # SQuAD
    tv.plot_em_quant({
                        'RoBERTa-linear': roberta_squad_quant_linear, \
                        'RoBERTa-linear-pruned': roberta_squad_quant_linear_clamped, \
                        'RoBERTa-log': roberta_squad_quant_log, \
                        'RoBERTa-log-pruned': roberta_squad_quant_clamped_log_1e3, \
                        'BERT-linear': bert_squad_quant_linear, \
                        'BERT-linear-pruned': bert_squad_quant_linear_clamped, \
                        'BERT-log': bert_squad_quant_log, 
                        'BERT-log-pruned': bert_squad_quant_log_clamped,
                        'RoBERTa-boolean': roberta_squad_quant_bin,
                        'BERT-boolean': bert_squad_quant_boolean,
                        }, 
                        ori_em={'RoBERTa': roberta_squad_original_em, 'BERT': bert_squad_ori_em}, \
                        ori_label_offset={'RoBERTa': [1.5, 1.7], 'BERT': [1.5, -2.5]},
                        ylabel='EM score', break_end=9.5, append_to_fname='_squad_midval')

    print('roberta squad log clamped:', (roberta_squad_original_em - roberta_squad_quant_clamped_log_1e3['em'][3.0]*100)/roberta_squad_original_em)
    print(roberta_squad_quant_clamped_log_1e3)
    print('bert squad log clamped:', (bert_squad_ori_em - bert_squad_quant_log_clamped['em'][3.0]*100)/bert_squad_ori_em)

    # MLM plot
    tv.plot_em_quant({'RoBERTa-linear': roberta_mlm_quant_linear, 
                        'RoBERTa-linear-pruned': roberta_mlm_quant_linear_clamped, 
                        'RoBERTa-log': roberta_mlm_quant_log, 
                        'RoBERTa-log-pruned': roberta_mlm_quant_log_clamped, 
                        'BERT-linear': bert_mlm_quant_linear, 
                        'BERT-linear-pruned': bert_mlm_quant_linear_clamped, 
                        'BERT-log': bert_mlm_quant_log,
                        'BERT-log-pruned': bert_mlm_quant_log_clamped
                        }, 
                        ori_em={'original': roberta_mlm_original_ppl, 'BERT': bert_mlm_original_ppl}, \
                        ori_label_offset={'original': [1.5, 1.7], 'BERT': [1.5, -2.5]},
                        break_start=15.9, break_end=9.5, ylabel='pseudo-perplexity',
                        append_to_fname='_mlm_midval', reverse_y=True, yscale='log', percent=False)
    
    print('roberta mlm log clamped:', (roberta_mlm_original_ppl - roberta_mlm_quant_log_clamped['em'][3.0])/roberta_mlm_original_ppl)
    print('bert mlm log clamped:', (bert_mlm_original_ppl - bert_mlm_quant_log_clamped['em'][3.0])/bert_mlm_original_ppl)

    # sst plot
    tv.plot_em_quant({'RoBERTa-linear': roberta_sst2_linear, 
                        'RoBERTa-linear-pruned': roberta_sst2_linear_clamped, 
                        'RoBERTa-log': roberta_sst2_log, 
                        'RoBERTa-log-pruned': roberta_sst2_log_clamped, 
                        # 'RoBERTa-uniform-log': roberta_sst2_uniform_slog_mean, 
                        # 'RoBERTa-uniform-log-pruned': roberta_sst2_uniform_slog_clamped_mean,
                        'RoBERTa-boolean': roberta_sst2_1bit},
                        ori_em={'RoBERTa': roberta_sst2_original_em},
                        ori_label_offset={'RoBERTa': [1.5, 1.2]}, 
                        break_end=8.5, append_to_fname='_sst_midval')
    print('roberta sst log clamped:', (roberta_sst2_original_em - roberta_sst2_log_clamped['em'][2.0]*100)/roberta_sst2_original_em)

    # tv.plot_em_quant({'RoBERTa-linear-asym': roberta_squad_hquant_linear, 'RoBERTa-even-log': roberta_squad_hquant_evenlog, \
    #                      'RoBERTa-even-log-smax': roberta_squad_hquant_evenlog_smax, \
    #                      'RoBERTa-fixed5': roberta_squad_hquant_fixed5, \
    #                      'RoBERTa-fixed4': roberta_squad_hquant_fixed4}, append_to_fname='_h_squad', fontsize=15)

    # comparing mid val quantization
    # mid val sst2
    # roberta_sst2_linear_midval = get_em_quantbits('./quantized_params/roberta-sst2-quant-linear-midval')
    # roberta_sst2_linear_clamped_midval = get_em_quantbits('./quantized_params/roberta-sst2-quant-linear-clamped-midval')
    # roberta_sst2_log_midval = get_em_quantbits('./quantized_params/roberta-sst2-quant-log-midval')
    # roberta_sst2_log_clamped_midval = get_em_quantbits('./quantized_params/roberta-sst2-quant-log-clamped-midval')

    # tv.plot_em_quant({'RoBERTa-uniform': roberta_sst2_linear_midval, 
    #                     'RoBERTa-uniform-clamped': roberta_sst2_linear_clamped_midval,
    #                     'RoBERTa-log': roberta_sst2_log_midval,
    #                     'RoBERTa-log-clamped': roberta_sst2_log_clamped_midval,
    #                     'RoBERTa-boolean': roberta_sst2_1bit}, 
    #                     ori_em={'RoBERTa': roberta_sst2_original_em}, append_to_fname='_sst_midval')

    # clamp threshold sweeping
    roberta_sst2_thres_sweep = get_em_quantbits('./quantized_params/roberta-sst2-sweep-thres-log-midval')
    roberta_mlm_thres_sweep = get_em_quantbits('./quantized_params/roberta-mlm-sweep-thres-log-midval')
    roberta_squad_thres_sweep = get_em_quantbits('./quantized_params/roberta-squad-sweep-thres-log-clamped-midval', avg_score=True)
    
    tv.plot_em_clamp_thres({'RoBERTa-SST': roberta_sst2_thres_sweep, 
                            'RoBERTa-SQuAD': roberta_squad_thres_sweep}, 
                        ori_em={'SST/MLM': roberta_sst2_original_em, 'SQuAD': roberta_squad_original_em},
                        ori_label_offset={'SST/MLM':[0.35, -4], 'SQuAD': [0.35, -4]}, 
                        second_axis_data={'RoBERTa-MLM': roberta_mlm_thres_sweep}
                        )
