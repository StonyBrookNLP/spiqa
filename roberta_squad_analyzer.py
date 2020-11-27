"""
roberta squad analyzer: analyzer sparsity of the roberta on squad 
"""

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import transformer_visualization as tv

import argparse as ag
import os
import sys
import random
import functools
import operator
from subprocess import call
from math import isnan, fsum, log
from textwrap import wrap
import urllib.request
import json
from itertools import compress, product

RES_FIG_PATH = "./res_fig/"
PARAM_PATH = "./params/"
DATA_PATH = "./data/"
FILT_PARAM_PATH = "./filtered_params/"
MAX_SEQ_LEN = 320
ATT_SIZE = [12, 12, MAX_SEQ_LEN, MAX_SEQ_LEN]
HS_SIZE = [ATT_SIZE[0]+1, 1, MAX_SEQ_LEN, 64*ATT_SIZE[1]]

def screen_clear():
    _ = call('clear' if os.name == 'posix' else 'cls', shell=True)


def parse_squad_json(squad_ver='v1.1'):
    FILE_PATH = DATA_PATH+"dev-"+squad_ver+".json"
    if not os.path.isfile(FILE_PATH):
        # download json file from web
        print("SQuAD {} file not found, try to download it...".format(squad_ver))
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-{}.json".format(
            squad_ver)
        data = (urllib.request.urlopen(url)).read()
        with open(FILE_PATH, "wb+") as out_file:
            out_file.write(data)

    data = {}
    with open(FILE_PATH, "r", encoding="utf-8") as data_file:
        squad_raw_data = json.load(data_file)["data"]

        for topic in squad_raw_data:
            for pgraph in topic["paragraphs"]:
                ques_per_paragraph = []
                for qa in pgraph["qas"]:
                    if (squad_ver == 'v1.1') or (squad_ver == "v2.0" and not qa["is_impossible"]):
                        gold_ans = [answer['text'] for answer in qa['answers']
                                    if normalize_answer(answer['text'])]
                        if not gold_ans:
                            gold_ans = [""]
                        ques_per_paragraph.append(
                            {"question": qa["question"], "answers": gold_ans})

                data[pgraph["context"]] = ques_per_paragraph

    return data

# def run_bert_wiki_pipeline():
#     wiki_pipeline = pipeline(
#         ""
#     )

def run_qa_pipeline(model_name: str, filter_inputs=True, single_input=True, sample_inputs=-1, att_threshold=0.0, hs_threshold=0.0):
    '''
    run question answering pipeline. 
    filter inputs: filter out the question-context pairs that have lengths out of 
    600-700 chars.
    sample inputs: randomly sample some question-context pairs to get all 
    raw attentions from each of them, instead of aggregrating the values
    the sample_inputs should be less than 100 to control the RAM usage under 5.89GB
    '''
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=0
    )

    print("Running pipeline...")
    data = parse_squad_json()
    associated_data = []
    for context in data.keys():
        context_ques_pair = []
        for ques in data[context]:
            context_ques_pair.append(
                {'context': context, 'question': ques['question'], 'answers': ques['answers']})
        associated_data.append(context_ques_pair)

    associated_data = sum(associated_data, [])
    input_lens = [len(i['context']+i['question']) for i in associated_data]
    print("QA string pair length: [{}, {}]".format(min(input_lens), max(input_lens)))

    # sample several instances from all data for short test
    if sample_inputs > 0:
        random.seed(123)
        associated_data = random.sample(associated_data, sample_inputs)

    fed_data = associated_data
    # construct and apply length filter to inputs
    # TODO: parameterize the length selector
    if filter_inputs:
        len_filter = [1 if 600 <= i < 700 else 0 for i in input_lens]
        filtered_associated_data = list(compress(associated_data, len_filter))
        fed_data = filtered_associated_data
    if single_input:
        single_associated_data = [random.choice(associated_data)]
        fed_data = single_associated_data
    
    # MARK: define head mask here
    head_mask = np.ones(ATT_SIZE[:2])
    head_mask[0][9], head_mask[0][11], head_mask[1][2], head_mask[7][8] = 0, 0, 0, 0
    
    res, pipeline_running_counter, fed_data_len = None, 0, len(fed_data)
    total_elem_count = 0
    print("Among all inputs {}/{} are selected.".format(fed_data_len, len(associated_data)))
    # run the prediction
    for qa_pair in fed_data:
        print("running pipeline iter {}/{}...".format(pipeline_running_counter, fed_data_len))
        prediction = qa_pipeline(
            {'context': qa_pair['context'], 'question': qa_pair['question']}, max_seq_len=MAX_SEQ_LEN, att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask)
        em_score = max(compute_exact(prediction['answer'], gold_ans)
                       for gold_ans in qa_pair['answers'])
        att_array = prediction['attentions']

        # aggregrate attention and hidden states
        # MARK: I am only getting values that are zero for the sparsity here. No specific sparsity bar.
        def get_spars(x, axis): 
            return x.shape[-1] ** 2 - np.count_nonzero(x[:, :, :x.shape[-1], :], axis=axis)
        def agg_func(f): return np.stack([f(i, axis=(-2, -1)) for i in att_array], axis=0)
        def add_func(f): return np.sum([f(i, axis=(-2, -1)) for i in att_array], axis=0)
        if res is None:
            res = {'score': em_score, 'hidden_states': np.zeros(HS_SIZE),
                   'max': agg_func(np.amax), 'min': agg_func(np.amin), 'mean': agg_func(np.mean),
                   'std': agg_func(np.std), 'sparsity': add_func(get_spars)}
            res['attentions'] = [] if sample_inputs > 0 else np.zeros(ATT_SIZE)
        else:
            res['score'] = (res['score'] + em_score)
            res['max'] = np.concatenate((res['max'], agg_func(np.amax)), axis=0)
            res['min'] = np.concatenate((res['min'], agg_func(np.amin)), axis=0)
            res['mean'] = np.concatenate((res['mean'], agg_func(np.mean)), axis=0)
            res['std'] = np.concatenate((res['std'], agg_func(np.std)), axis=0)
            res['sparsity'] = np.add(res['sparsity'], add_func(get_spars))

        # collect attentions
        if sample_inputs > 0:
            res['attentions'] += att_array
            if np.count_nonzero(res['hidden_states']) == 0: res['hidden_states'] = prediction['hidden_states']
            else: res['hidden_states'] = np.concatenate((res['hidden_states'], prediction['hidden_states']), axis=1)
        else:
            for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['hidden_states'], prediction['hidden_states'])):
                res['hidden_states'][layer_idx][0] = np.add(res_layer[0], pred_layer[0])
            for att in att_array:
                padded_att = np.zeros(ATT_SIZE)
                padded_att[:, :, :att.shape[2], :att.shape[3]] = att
                # aggregrate all the results
                # unfold the tensor to 2-D array to walk around buggy numpy sum
                for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['attentions'], padded_att)):
                    for head_idx, (res_head, pred_head) in enumerate(zip(res_layer, pred_layer)):
                        res['attentions'][layer_idx][head_idx] = np.add(res_head, pred_head)

        pipeline_running_counter += 1
        total_elem_count += sum([att.shape[-1] * att.shape[-1] for att in att_array])

        if (sample_inputs > 0): 
            for i in res['attentions']: 
                if (i > len(res['attentions'])).any() :
                    idx0, idx1, idx2, idx3, idx4 = np.where(i > len(res['attentions']))
                    print("iter {} has attention larger than 1 ({}), exist..."
                        .format(len(res['attentions']), (idx0[0], idx1[0], idx2[0], idx3[0], idx4[0])))
                    exit()

        print(prediction['answer'], em_score, res['score'] / pipeline_running_counter)

    res['sparsity'] = res['sparsity'].astype(float) / total_elem_count
    res['qa_pair_len'] = fed_data_len
    return res


def get_hstates_attens(model_name: str, force_reinfer=False, filter_inputs=True, single_input=True, sample_inputs=-1, layer_aggregration='mean', att_threshold=0.0, hs_threshold = 0.0):
    '''
    get the hidden state and attention from pipeline result. 
    The model_name should be a valid Huggingface transformer model. 
    Enable force_reinfer if one wants to ignore the existing npy file 
    and re-do the inference anyway.
    filter inputs: filter out the question-context pairs that have lengths out of 
    600-700 chars.
    sample inputs: randomly sample some question-context pairs to get all 
    raw attentions from each of them, instead of aggregrating the values
    the sample_inputs should be less than 100 to control the RAM usage under 5.89GB
    sample inputs is -1 means using all inputs
    '''
    all_hidden_states, all_attentions, total_score, qa_pair_count, \
        all_max, all_min, all_mean, all_std, all_sparsity = \
        None, None, None, None, None, None, None, None, None

    if sample_inputs > 100 or sample_inputs == 0:
        raise ValueError("the sample inputs should be (0, 100]")
    # read from file
    input_type = "_sampled" if sample_inputs > 0 else "_all"
    input_type += "_filtered" if filter_inputs else ""
    h_states_path, atten_path, score_path, att_stat_path = \
        (PARAM_PATH + i + input_type +
         '.npy' for i in ['hidden_states', 'attentions', 'score', 'att_stat_features'])
    if os.path.isfile(h_states_path) and os.path.isfile(atten_path) and \
            os.path.isfile(score_path) and os.path.isfile(att_stat_path) and not force_reinfer:
        print("Loading parameters from file {}...".format(PARAM_PATH + input_type))
        with open(score_path, "rb") as score_file:
            total_score, qa_pair_count = (i for i in np.load(score_file))
        with open(h_states_path, "rb") as h_states_file:
            all_hidden_states = np.load(h_states_file)
        with open(atten_path, "rb") as attention_file:
            if sample_inputs > 0:
                atten_len, all_attentions = (np.load(attention_file))[0], []
                for i in range(atten_len): all_attentions.append(np.load(attention_file))
            else: all_attentions = np.load(attention_file)
        with open(att_stat_path, "rb") as att_stat_file:
            all_max = np.load(att_stat_file)
            all_min = np.load(att_stat_file)
            all_mean = np.load(att_stat_file)
            all_std = np.load(att_stat_file)
            all_sparsity = np.load(att_stat_file)
    # extract parameters from model
    else:
        print("Extracting attentions from model...")
        predictions = run_qa_pipeline(
            model_name, filter_inputs=filter_inputs, single_input=single_input, \
            sample_inputs=sample_inputs, att_threshold=att_threshold, hs_threshold=hs_threshold)

        total_score, all_hidden_states, all_attentions, qa_pair_count, \
            all_max, all_min, all_mean, all_std, all_sparsity = \
            predictions['score'], predictions['hidden_states'], \
            predictions['attentions'], predictions['qa_pair_len'], \
            predictions['max'], predictions['min'], predictions['mean'], predictions['std'], predictions['sparsity']

        with open(score_path, "wb+") as scores_file:
            np.save(scores_file, np.array([total_score, qa_pair_count]))
        with open(h_states_path, "wb+") as h_states_file:
            np.save(h_states_file, all_hidden_states)
        with open(atten_path, "wb+") as attention_file:
            if sample_inputs > 0:
                np.save(attention_file, np.array([len(all_attentions)]))
                for i in all_attentions: np.save(attention_file, i)
            else: np.save(attention_file, all_attentions)
        with open(att_stat_path, "wb+") as att_stat_file:
            np.save(att_stat_file, all_max)
            np.save(att_stat_file, all_min)
            np.save(att_stat_file, all_mean)
            np.save(att_stat_file, all_std)
            np.save(att_stat_file, all_sparsity)

    print("total score: ", total_score, "#QA pair: ", qa_pair_count,
          "hidden_state dim: ", all_hidden_states.shape,
          "max dim:", all_max.shape, "min dim:", all_min.shape,
          "mean dim:", all_mean.shape, "std dim:", all_std.shape,
          "sparsity dim:", all_sparsity.shape)
    if sample_inputs > 0:
        print("attention dim:", len(all_attentions), all_attentions[0].shape)
    else:
        print("attention dim:", all_attentions[0].shape)

    total_score /= float(qa_pair_count)

    if layer_aggregration == 'mean' and sample_inputs == 0:
        all_hidden_states /= float(qa_pair_count)
        all_attentions /= float(qa_pair_count)

    return total_score, all_hidden_states, all_attentions, all_max, all_min, all_mean, all_std, all_sparsity


def get_sparsities(params_path: str, sparsity_bar=0.025, layer_aggregration='mean'):
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
        sparsity_table.at[threshold, 'em'] = total_score

    return sparsity_table


def get_stat_features(att_features: dict):
    '''
    get pandas dataframe of min, max, mean, avg and std of 12 layers x 12 heads accross all instances
    '''
    stat_tab_idx = ["L{}H{}".format(
        i, j) for i, j in list(product(range(0, 12), range(0, 12)))]
    stat_table = pd.DataFrame(index=stat_tab_idx)
    stat_func_list = {'min': np.amin, 'max': np.amax, 'avg': np.average, 'std': np.std}
    for att_feature_key in att_features.keys():
        for stat_func in stat_func_list.keys():
            stat_table['{}_{}'.format(stat_func, att_feature_key)] = \
                stat_func_list[stat_func](
                    att_features[att_feature_key], axis=0).flatten().tolist()

    return stat_table


def plot_dist(data, bin_step, sparsity_bar=0.025, single_head_idx=None, layer_aggregration='mean', attached_title=''):
    '''
    Plot the histrogram to visualize the distribution of the self attention 
    matrix for each attention head in each layer.

    expected data shape: (#layers, #insts, #heads, length, length)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    '''
    # set histogram x axis starting point here
    offset = 1e-8
    hist_x_start, hist_x_end = log(offset, 10), log(1+offset, 10)

    def get_bin_edges(bin_step, head_idx, layer_idx, scale='normal'):
        if type(bin_step) is int:
            if scale == 'log':
                bin_edges = 10**np.linspace(hist_x_start, hist_x_end, bin_step+1)
                bin_edges[0] -= 10**(hist_x_start-1)
                return pd.Series(bin_edges)
            else:
                return bin_step
        elif type(bin_step) is float:
            if scale == 'log':
                bin_edges = 10**np.append(
                    np.arange(hist_x_start, hist_x_end, bin_step), hist_x_end)
                bin_edges[0] -= 10**(hist_x_start-1)
                return pd.Series(bin_edges)
            else:
                return pd.Series(np.append(np.arange(0, 1.0, bin_step), 1.0))
        elif type(bin_step) is list:
            return pd.Series(np.append(np.arange(0.0, 1.0, bin_step[layer_idx][head_idx]), 1.0))
        else:
            return None

    if single_head_idx is None:
        # walk through layers and heads
        # data = np.concatenate(data, axis=-1)
        # data = data.reshape(*data.shape[:2], -1)
        # print(data.shape)
        
        for layer_idx, layer in enumerate(data):
            print('plotting histogram for layer {}...'.format(layer_idx))
            atten_layers = {}
            for head_idx, head in enumerate(layer):
                sparsity = (head <= (sparsity_bar)).sum() // head.flatten().shape[0]
                atten_layers['head_{}, max: {:.4f}, min: {:.4f}, spars: {:.4f}, sparsity_bar: {:.4f}'.format(
                    head_idx, torch.max(head), torch.min(head), sparsity, sparsity_bar)] = (head.flatten()+offset).tolist()

            atten_layers_pd = pd.DataFrame(atten_layers)
            # create vars for plotting
            fig, ax = plt.subplots(3, 4, figsize=(21, 12))
            # extract pd column name and column into head_idx and head respectively
            for head_idx, head in atten_layers_pd.iteritems():
                head_idx_int = int(head_idx.split(',')[0].split('_')[1])
                curr_ax = ax[int(head_idx_int/4), int(head_idx_int % 4)]
                head.hist(ax=curr_ax, bins=get_bin_edges(bin_step, layer_idx, head_idx, scale='log'),
                          weights=(np.ones_like(head) / len(head)), color='C0')
                head.hist(ax=curr_ax, bins=get_bin_edges(bin_step, layer_idx, head_idx, scale='log'),
                          weights=(np.ones_like(head) / len(head)), cumulative=True,
                          histtype='step', linewidth=1, color='C3')
                curr_ax.set_xscale('log')
                curr_ax.set_xlim([10 ** hist_x_start, 10 ** hist_x_end])
                # set y as log as well
                # curr_ax.set_yscale('log')
                curr_ax.set_ylim([0.0, 1.0])
                curr_ax.set_title('\n'.join(wrap(head_idx, 38)))

            for axis in ax.flatten():
                axis.grid(linestyle='--', color='grey', alpha=0.6)
            fig.suptitle(
                'Histogram of Layer {}\'s Attention per head (batch aggregation={}, {})'
                .format(layer_idx, layer_aggregration, attached_title), fontsize=21, y=0.99)
            fig.tight_layout()
            plt.savefig(RES_FIG_PATH+'hist_layer{}.png'.format(layer_idx), dpi=600)
            plt.clf()
            plt.close(fig)
    elif type(single_head_idx) is tuple and len(single_head_idx) == 2:
        layer_idx, head_idx = single_head_idx
        head = data[layer_idx][0][head_idx].flatten()
        sparsity = (head <= (sparsity_bar)).sum() / head.shape[0]
        head = pd.Series(head)
        fig, ax = plt.subplots(figsize=(20, 6))
        head.hist(ax=ax, bins=get_bin_edges(bin_step, layer_idx, head_idx),
                  weights=(np.ones_like(head) / len(head)), color='C0')
        head.hist(ax=ax, bins=get_bin_edges(bin_step, layer_idx, head_idx),
                  weights=(np.ones_like(head) / len(head)), cumulative=True,
                  histtype='step', linewidth=1, color='C3')
        ax.set_title('layer_{}_head_{}, max: {:.4f}, min: {:.4f}, spars: {:.4f}, sparsity_bar: {:.4f}'
                     .format(layer_idx, head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar))
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.grid(linestyle='--', color='grey', alpha=0.6)
        fig.tight_layout()
        plt.savefig(
            RES_FIG_PATH+'hist_singlehead_layer{}_head{}.png'.format(layer_idx, head_idx), dpi=600)
        plt.clf()
        plt.close(fig)


def plot_dist_token_dynamic(model_name, bin_step, sparsity_bar=0.025, att_threshold=0.0, attached_title='', samples=10, scale='log'):
    '''
    computing histogram per token on-the-fly without saving the attentions in the memory
    '''
    # set histogram x axis starting point here
    offset = 1e-12
    hist_x_start, hist_x_end = log(offset, 10), log(1+offset, 10)
    if scale == 'linear':
        offset = 0.0
    qa_pipeline = pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=0
    )

    def get_bin_edges(bin_step):
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

    file_type = "_sampled_per_token"
    hist_file_path = PARAM_PATH + "atten_hist{}.npy".format(file_type)

    atten_bins, atten_hist, all_score = get_bin_edges(bin_step), None, 0
    all_max, all_min, all_sparse_count, sparse_hist, all_seq_len = None, None, None, None, None

    if os.path.isfile(hist_file_path):
        print("loading histogram from ", hist_file_path)
        with open(hist_file_path, "rb") as hist_file:
            atten_hist = np.load(hist_file)
            atten_bins = np.load(hist_file)
            all_seq_len = np.load(hist_file)
            all_max = np.load(hist_file)
            all_min = np.load(hist_file)
            all_sparse_count = np.load(hist_file)
            sparse_hist = np.load(hist_file)
    else:
        print("Running pipeline...")
        data = parse_squad_json()
        associated_data = []
        for context in data.keys():
            context_ques_pair = []
            for ques in data[context]:
                context_ques_pair.append(
                    {'context': context, 'question': ques['question'], 'answers': ques['answers']})
            associated_data.append(context_ques_pair)
        
        # fixed random seed to select same subsets of the instances every time for comparison
        random.seed(123)
        associated_data = random.sample(sum(associated_data, []), samples)
        input_lens = [len(i['context']+i['question']) for i in associated_data]
        print("QA string pair length: [{}, {}]".format(min(input_lens), max(input_lens)))
        pipeline_running_counter, fed_data_len = 0, len(associated_data)

        # MARK: define head mask here
        head_mask = np.ones(ATT_SIZE[:2])
        head_mask[0][9], head_mask[0][11], head_mask[1][2], head_mask[7][8] = 0, 0, 0, 0

        # run the prediction, calculate and store the hist
        for qa_pair in associated_data:
            print("running pipeline iter {}/{}...".format(pipeline_running_counter, fed_data_len))
            prediction = qa_pipeline(
                {'context': qa_pair['context'], 'question': qa_pair['question']}, max_seq_len=320, att_threshold=att_threshold, head_mask=head_mask)
            pipeline_running_counter += 1
            em_score = max(compute_exact(prediction['answer'], gold_ans)
                           for gold_ans in qa_pair['answers'])

            for att in prediction['attentions']:
                att = att[:, :, :att.shape[-1], :]
                print("min:", np.amin(att[att.nonzero()]))
                curr_hist = np.apply_along_axis(lambda a: np.histogram(a+offset, atten_bins, range=(0.0, 1.0))[0], -1, att)
                atten_hist = [curr_hist] if atten_hist is None else atten_hist + [curr_hist]
                curr_sparse_count = np.apply_along_axis(lambda a: float((a <= sparsity_bar).sum()) / att.shape[-1], -1, att)
                all_sparse_count = curr_sparse_count if all_sparse_count is None \
                                    else np.concatenate((curr_sparse_count, all_sparse_count), axis=-1)
                all_seq_len = [att.shape[-1]] if all_seq_len is None else all_seq_len + [att.shape[-1]]
                curr_max, curr_min = np.amax(att, axis=(-2, -1)), np.amin(att, axis=(-2, -1))

            all_score += em_score
            all_max = curr_max if all_max is None else np.maximum(all_max, curr_max)
            all_min = curr_min if all_min is None else np.minimum(all_min, curr_min)

        atten_hist = np.concatenate(atten_hist, axis=-2)
        sparse_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=10, range=(0.0, 1.0))[0], -1, all_sparse_count)
        
        print("atten_hist shape:", atten_hist.shape)
        print("sparsity shape:", all_sparse_count.shape)
        print("sparsity hist shape:", sparse_hist.shape)
        print("all seq shape:", len(all_seq_len))
        print("EM score", all_score / fed_data_len)

        # Normalization
        atten_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, atten_hist)
        sparse_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, sparse_hist)
        # save the histogram
        with open(hist_file_path, "wb+") as hist_file:
            np.save(hist_file, atten_hist, allow_pickle=False)
            np.save(hist_file, atten_bins, allow_pickle=False)
            np.save(hist_file, all_seq_len, allow_pickle=False)
            np.save(hist_file, all_max, allow_pickle=False)
            np.save(hist_file, all_min, allow_pickle=False)
            np.save(hist_file, all_sparse_count, allow_pickle=False)
            np.save(hist_file, sparse_hist, allow_pickle=False)

    # plot atten_hist
    tv.plot_atten_dist_per_token(atten_hist, bin_step, all_max, all_min, sparse_hist=sparse_hist)

    # plot sparsity histogram when sampling:
    # if samples > 0:
    #     for layer_idx, layer in enumerate(atten_hist):
    #         fig, ax = plt.subplots(3, 4, figsize=(21, 12))
    #         for head_idx, head in enumerate(layer):
    #             curr_ax = ax[int(head_idx/4), int(head_idx % 4)]
    #             head_sparsity = all_sparse_count[layer_idx][head_idx]
    #             curr_ax.hist(head_sparsity, bins=100, range=(0, 1.0), weights=(
    #                 np.ones_like(head_sparsity)/len(head_sparsity)))
    #             curr_ax.hist(head_sparsity, bins=100, range=(0, 1.0), weights=(np.ones_like(head_sparsity)/len(head_sparsity)),
    #                          cumulative=True, histtype='step', linewidth=1, color='C3')

    #             subplot_title = 'head_{}, max: {:.4f}, min: {:.4f}'.format(
    #                 head_idx, all_max[layer_idx][head_idx], all_min[layer_idx][head_idx])

    #             curr_ax.set_title('\n'.join(wrap(subplot_title, 38)))
    #             curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
    #             curr_ax.set_ylim([0, 1])
    #             curr_ax.set_xlim([0, 1])

    #         fig.suptitle("Sparsity Histogram for layer {} per head {}, with sparsity bar {:.4f}".format(
    #             layer_idx, attached_title, sparsity_bar), fontsize=21, y=0.99)
    #         fig.tight_layout()
    #         plt.savefig(
    #             RES_FIG_PATH+'spars_hist_layer_otf_{}{}.png'.format(layer_idx, file_type), dpi=600)
    #         plt.clf()
    #         plt.close(fig)


def plot_sparsity_change(data, attached_title=''):
    '''
    plot sparsity change for different sparsity dropout threshold
    '''
    att_threshold = [float(i) for i in data.index.tolist()]
    for layer_idx in range(0, 12):
        print('plotting curve for sparsities...')
        fig, ax = plt.subplots(3, 4, figsize=(21, 12))
        for head_idx in range(0, 12):
            curr_ax = ax[int(head_idx/4), int(head_idx % 4)]
            curr_ax.plot(att_threshold, data['layer_{}_head_{}'.format(
                layer_idx, head_idx)].tolist(), color='C0', marker='s')
            curr_ax.set_title('head {}'.format(head_idx))
            curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
            curr_ax.set_ylim([0.0, 1.01])
            curr_ax.set_xlim(0.0, max(att_threshold)+0.01)

        fig.suptitle('Sparsity for Different Thresholds for Layer {} {}'.format(
            layer_idx, attached_title), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(RES_FIG_PATH+'spars_change_layer{}.png'.format(layer_idx), dpi=600)
        plt.clf()
        plt.close(fig)

    # plot sparsity/accu vs threshold
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6)

    # legends
    patches = []
    patches.append(mpatches.Patch(color='C0', label='sparsity'))
    patches.append(mpatches.Patch(color='C1', label='EM score'))

    ax1.set_xlabel('sparsity dropping threshold')
    ax1.set_ylabel('sparsity')
    ax1.plot(att_threshold, data['all'], color='C0', marker='s', markersize='4.5')
    ax2 = ax1.twinx()
    ax2.set_ylabel('EM score')
    ax2.plot(att_threshold, data['em']*100, color='C1', marker='s', markersize='4.5')
    ax1.set_yticks(np.linspace(0.2, 1.1, 10))
    ax1.set_ylim([0.2, 1.1])
    ax2.set_yticks(np.linspace(20, 110, 10))
    ax2.set_ylim([20, 110])

    ax2.set_xscale('linear')
    ax2.set_xlim([0, 1.2])
    fig.suptitle(
        'Sparsity and Accuracy vs. Sparsity Dropping Threshold {}'.format(attached_title))
    fig.tight_layout()
    plt.grid(linestyle='--', alpha=0.5, color='grey')
    plt.legend(handles=patches, loc='upper left')
    plt.savefig(RES_FIG_PATH+'sparse_accu.png', dpi=600)
    plt.close(fig)


def plot_em_sparsity(sparsity_data: dict, attached_title=''):
    # plot em vs. sparsity
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    patches = []

    ax.set_xlabel("sparisty")
    ax.set_ylabel("pseudo-perplexity")

    for idx, (data_label, data) in enumerate(sparsity_data.items()):
        patches.append(mpatches.Patch(color='C{}'.format(idx), label=data_label))
        ax.plot(data['all'], data['em'],
                color='C{}'.format(idx), marker='s', markersize=4.5)

    # ax.set_ylim([30, 90])
    # ax.set_yscale('log')
    fig.suptitle(
        'Accuracy vs. Sparsity {}'.format(attached_title))
    fig.tight_layout()
    plt.legend(handles=patches, loc='upper left')
    plt.grid(linestyle='--', alpha=0.5, color='grey')
    plt.savefig(RES_FIG_PATH+'perplexity_vs_sparsity.png', dpi=600)
    plt.close(fig)


def plot_stat_features(stat_features, features_to_plot=['max', 'min', 'std']):
    num_features = len(features_to_plot)
    fig, ax = plt.subplots(num_features, 1, figsize=(24, num_features*4), sharex=True)
    for i, stat_feature in enumerate(features_to_plot):
        means, stds, maxs, mins = stat_features['avg_{}'.format(stat_feature)], \
            stat_features['std_{}'.format(stat_feature)], \
            stat_features['max_{}'.format(stat_feature)], \
            stat_features['min_{}'.format(stat_feature)]
        ax[i].errorbar(stat_features.index, means, yerr=[means - mins, maxs - means],
                       fmt='.', ecolor='grey', capsize=3, lw=1)
        ax[i].errorbar(stat_features.index, means, yerr=stds, fmt='ok', lw=3)
        ax[i].grid(linestyle='--', color='grey', alpha=0.4)
        ax[i].margins(0.002)
        ax[i].set_title('{}'.format(stat_feature), fontsize=18)
        for l in range(0, 12):
            ax[i].axvspan(l*12-0.5, l*12+12-0.5, alpha=0.2, facecolor='C{}'.format(l))

    plt.xticks(rotation=60)
    fig.suptitle('Statistical Features for Layers of Heads', fontsize=21, y=0.99)
    fig.tight_layout()
    fig.savefig(RES_FIG_PATH + 'stat_features.png', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    arg_parser = ag.ArgumentParser(description=__doc__)
    arg_parser.add_argument("-at", "--att_threshold", default=0.0,
                            required=False, help="set attention sparsity threshold")
    arg_parser.add_argument("-ht", "--hs_threshold", default=0.0,
                            required=False, help="set hidden states sparsity threshold")
    arg_parser.add_argument("-d", "--distribution", default=False, action='store_true',
                            required=False, help="print histogram")
    arg_parser.add_argument("-e", "--evaluation", default=False, action="store_true",
                            required=False, help="evaluate model only without any plot")
    arg_parser.add_argument("-m", "--heatmap", default=False, action="store_true",
                            required=False, help="print heatmap")
    arg_parser.add_argument("-s", "--sparsity", default=False, action='store_true',
                            required=False, help="compute sparsity")
    arg_parser.add_argument("-od", "--otf_distribution", default=False, action='store_true',
                            required=False, help='print attention histogram without saving aggregrated params')
    arg_parser.add_argument("-hs", "--hidden_states", default=False, action='store_true',
                            required=False, help='print hidden states histogram without saving aggregrated params')
    arg_parser.add_argument("-sa", "--samples", default=-1,
                            required=False, help="number of samples for distribution")

    args = vars(arg_parser.parse_args())
    att_threshold = float(args['att_threshold'])
    hs_threshold = float(args['hs_threshold'])
    samples = int(args['samples'])

    if args['evaluation']:
        em_score, h_states, attens, att_max, att_min, att_mean, att_std, att_sparsity = \
            get_hstates_attens("csarron/roberta-base-squad-v1", filter_inputs=False, force_reinfer=False,
                               single_input=False, layer_aggregration='mean', att_threshold=att_threshold, hs_threshold=hs_threshold, sample_inputs=samples)
        em_str = 'EM={:.2f}'.format(em_score*100)

    if args['distribution']:
        em_score, h_states, attens, att_max, att_min, att_mean, att_std, att_sparsity = \
            get_hstates_attens("csarron/roberta-base-squad-v1", filter_inputs=False, force_reinfer=False,
                               single_input=False, layer_aggregration='mean', att_threshold=att_threshold, hs_threshold=hs_threshold, sample_inputs=samples)
        em_str = 'EM={:.2f}'.format(em_score*100)
        stat_features = get_stat_features(
            {'max': att_max, 'min': att_min, 'mean': att_mean, 'std': att_std})
        print(stat_features)
        plot_stat_features(stat_features)
        stat_features.to_csv('stat_features_unfiltered.csv', sep=',')

        # plot histogram for all layers and all heads
        plot_dist(attens, bin_step=100, sparsity_bar=0.0005,
                  layer_aggregration='None', attached_title=em_str)
        # # plot histogram for a certain head in a certain layer
        # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
        #           single_head_idx=(0, 0), attached_title=em_str)
        # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
        #           single_head_idx=(0, 9), attached_title=em_str)
        # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
        #           single_head_idx=(0, 11), attached_title=em_str)

        # only plot heatmaps when distribution is available, temperarily broken
        if args['heatmap']:
            # plot heatmaps
            tv.plot_heatmap(attens, sparsity_bar=0.0005, binarize=False, attached_title=em_str)
            tv.plot_heatmap(attens, sparsity_bar=0.0005, binarize=True, attached_title=em_str)
            tv.plot_heatmap(attens, sparsity_bar=0.0005, binarize=False,
                        auto_scale=True, attached_title=em_str)

    if args['sparsity']:
        # compute sparsity, temperarily broken
        bert_uncased_spars = get_sparsities('filtered_params/bert-base-mlm')
        roberta_spars = get_sparsities('filtered_params/roberta-base-mlm')
        print(bert_uncased_spars, roberta_spars)
        plot_em_sparsity({'BERT': bert_uncased_spars, 'RoBERTa': roberta_spars})
        plot_sparsity_change(roberta_spars, attached_title='')

    if args['otf_distribution']:
        plot_dist_token_dynamic("csarron/roberta-base-squad-v1", 100, sparsity_bar=0.0, att_threshold=att_threshold, samples=samples, scale='log', attached_title='(per_token)')

    if args['hidden_states']:
        em_score, h_states, attens, att_max, att_min, att_mean, att_std, att_sparsity = \
            get_hstates_attens("csarron/roberta-base-squad-v1", filter_inputs=False, force_reinfer=False,
                               single_input=False, layer_aggregration='mean', att_threshold=att_threshold, hs_threshold=hs_threshold, sample_inputs=samples)
        attn_mask = [i.shape[-1] for i in attens]
        # h_state sanity check
        # for i in range(10):
        #     print("h_state mean:{:.4f}, std:{:.4f}".format(
        #         np.mean(h_states[0][0][i*5], axis=-1), np.std(h_states[0][0][i*5], axis=-1)))
        tv.plot_hs_dist_per_token(h_states, 100, attn_mask, scale='linear')
