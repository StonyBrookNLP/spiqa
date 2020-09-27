from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import argparse
import os
import random
from subprocess import call
from math import isnan, fsum
from textwrap import wrap
import urllib.request
import json
from itertools import compress, product

RES_FIG_PATH = "./res_fig/"
PARAM_PATH = "./params/"
DATA_PATH = "./data/"
FILT_PARAM_PATH = "./filtered_params/"


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


def run_qa_pipeline(model_name: str, filter_inputs=True, single_input=True):
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
    # construct and apply length filter to inputs
    # TODO: parameterize the length selector
    len_filter = [1 if 600 <= i < 700 else 0 for i in input_lens]
    filtered_associated_data = list(compress(associated_data, len_filter))
    single_associated_data = [random.choice(associated_data)]
    associated_data = random.sample(associated_data, 3)
    fed_data = filtered_associated_data if filter_inputs else associated_data
    fed_data = single_associated_data if single_input else fed_data

    res, pipeline_running_counter, fed_data_len = None, 0, len(fed_data)
    print("Among all inputs {}/{} are selected.".format(fed_data_len, len(associated_data)))
    # run the prediction
    for qa_pair in fed_data:
        print("running pipeline iter {}/{}...".format(pipeline_running_counter, fed_data_len))
        prediction = qa_pipeline(
            {'context': qa_pair['context'], 'question': qa_pair['question']}, max_seq_len=320)
        em_score = max(compute_exact(prediction['answer'], gold_ans)
                       for gold_ans in qa_pair['answers'])
        att_array = prediction['attentions']

        # aggregrate attention and hidden states
        if res is None:
            res = {'score': em_score, 'hidden_states': prediction['hidden_states'], 'attentions': att_array, 
                   'max': np.amax(att_array.reshape(*att_array.shape[:3], -1), axis=-1), 
                   'min': np.amin(att_array.reshape(*att_array.shape[:3], -1), axis=-1), 
                   'std': np.std(att_array.reshape(*att_array.shape[:3], -1), axis=-1)}
        else:
            res['score'] = (res['score'] + em_score)
            res['max'] = np.fmax(res['max'], np.amax(att_array.reshape(*att_array.shape[:3], -1), axis=-1))
            res['min'] = np.fmin(res['min'], np.amin(att_array.reshape(*att_array.shape[:3], -1), axis=-1))
            res['std'] = np.concatenate((res['std'], np.std(att_array.reshape(*att_array.shape[:3], -1), axis=-1)), axis=1)
            # unfold the tensor to 2-D array to walk around buggy numpy sum
            for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['hidden_states'], prediction['hidden_states'])):
                res['hidden_states'][layer_idx][0] = np.add(res_layer[0], pred_layer[0])

            for layer_idx, (res_layer, pred_layer) in enumerate(zip(res['attentions'], prediction['attentions'])):
                for head_idx, (res_head, pred_head) in enumerate(zip(res_layer[0], pred_layer[0])):
                    res['attentions'][layer_idx][0][head_idx] = np.add(
                        res_head, pred_head)

        pipeline_running_counter += 1

        if ((res['attentions'] > pipeline_running_counter).any()):
            idx0, idx1, idx2, idx3, idx4 = np.where(
                res['attentions'] > pipeline_running_counter)
            print("iter {} has attention larger than 1 ({}), exist..."
                  .format(pipeline_running_counter, (idx0[0], idx1[0], idx2[0], idx3[0], idx4[0])))
            exit()

        print(prediction['answer'], em_score, res['score'] / pipeline_running_counter)

        # check sparsity filter apply
        sampled_data = prediction['attentions'] * (prediction['attentions'] <= 0.005)
        if (sampled_data > 0.0).any():
            print("sparsity check failed!")
            exit()

        # screen_clear()

    res['qa_pair_len'] = fed_data_len
    return res


def get_hstates_attens(model_name: str, force_reinfer=False, filter_inputs=True, single_input=True, layer_aggregration='mean'):
    '''
    get the hidden state and attention from pipeline result. 
    The model_name should be a valid Huggingface transformer model. 
    Enable force_reinfer if one wants to ignore the existing npy file 
    and re-do the inference anyway.
    '''
    all_hidden_states, all_attentions, total_score, qa_pair_count, all_max, all_min, all_std = \
        None, None, None, None, None, None, None
    # read from file
    input_type = "_filtered" if filter_inputs else "_all"
    h_states_path, atten_path, score_path, att_stat_path = \
        (PARAM_PATH + i + input_type +
         '.npy' for i in ['hidden_states', 'attentions', 'score', 'att_stat_features'])
    if os.path.isfile(h_states_path) and os.path.isfile(atten_path) and \
            os.path.isfile(score_path) and os.path.isfile(att_stat_path) and not force_reinfer:
        print("Loading parameters from file...")
        with open(score_path, "rb") as score_file:
            total_score, qa_pair_count = (i for i in np.load(score_file))
        with open(h_states_path, "rb") as h_states_file:
            all_hidden_states = np.load(h_states_file)
        with open(atten_path, "rb") as attention_file:
            all_attentions = np.load(attention_file)
        with open(att_stat_path, "rb") as att_stat_file:
            all_max = np.load(att_stat_file)
            all_min = np.load(att_stat_file)
            all_std = np.load(att_stat_file)

    # extract parameters from model
    else:
        print("Extracting attentions from model...")
        predictions = run_qa_pipeline(
            model_name, filter_inputs=filter_inputs, single_input=single_input)

        total_score, all_hidden_states, all_attentions, qa_pair_count, all_max, all_min, all_std = \
            predictions['score'], predictions['hidden_states'], \
            predictions['attentions'], predictions['qa_pair_len'], \
            predictions['max'], predictions['min'], predictions['std']

        with open(score_path, "wb+") as scores_file:
            np.save(scores_file, np.array([total_score, qa_pair_count]))
        with open(h_states_path, "wb+") as h_states_file:
            np.save(h_states_file, all_hidden_states)
        with open(atten_path, "wb+") as attention_file:
            np.save(attention_file, all_attentions)
        with open(att_stat_path, "wb+") as att_stat_file:
            np.save(att_stat_file, all_max)
            np.save(att_stat_file, all_min)
            np.save(att_stat_file, all_std)

    print("total score: ", total_score, "#QA pair: ", qa_pair_count,
          "hidden_state dim: ", all_hidden_states.shape,
          "attention dim:", all_attentions.shape,
          "max dim:", all_max.shape, "min dim:", all_min.shape, "std dim:", all_std.shape)

    if layer_aggregration == 'mean':
        total_score /= float(qa_pair_count)
        all_hidden_states /= float(qa_pair_count)
        all_attentions /= float(qa_pair_count)

    return total_score, all_hidden_states, all_attentions


def get_sparsities(threshold_list: list, sparsity_bar=0.025, layer_aggregration='mean'):
    '''
    extract sparsities for a fixed sparsity bar from all parameters with different threshold.
    '''
    params_path_list = [FILT_PARAM_PATH +
                        i.replace('.', '_') + "/" for i in threshold_list]
    sparsity_table = pd.DataFrame(index=[i for i in threshold_list])

    for threshold, params in zip(threshold_list, params_path_list):
        all_hidden_states, all_attentions, total_score, qa_pair_count = None, None, None, None
        # read from file
        input_type = "_all"
        h_states_path, atten_path, score_path = \
            (params + i + input_type +
             '.npy' for i in ['hidden_states', 'attentions', 'score'])
        if os.path.isfile(h_states_path) and os.path.isfile(atten_path) and \
                os.path.isfile(score_path):
            with open(score_path, "rb") as score_file:
                total_score, qa_pair_count = (i for i in np.load(score_file))
            with open(h_states_path, "rb") as h_states_file:
                all_hidden_states = np.load(h_states_file)
            with open(atten_path, "rb") as attention_file:
                all_attentions = np.load(attention_file)

        if layer_aggregration == 'mean':
            total_score /= float(qa_pair_count)
            all_hidden_states /= float(qa_pair_count)
            all_attentions /= float(qa_pair_count)

        for layer_idx, layer in enumerate(all_attentions):
            for head_idx, head in enumerate(layer[0]):
                sparsity = (head <= sparsity_bar).sum() / head.flatten().shape[0]
                sparsity_table.at[threshold, 'layer_{}_head_{}'.format(
                    layer_idx, head_idx)] = sparsity

    return sparsity_table


def plot_dist(data, bin_step, sparsity_bar=0.025, single_head_idx=None, layer_aggregration='mean', attached_title=''):
    '''
    Plot the histrogram to visualize the distribution of the self attention 
    matrix for each attention head in each layer.

    expected data shape: (#layers, #heads, length, dv)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    '''
    def get_bin_edges(bin_step, head_idx, layer_idx):
        if type(bin_step) is int:
            return bin_step
        elif type(bin_step) is float:
            return pd.Series(np.append(np.arange(0.0, 1.0, bin_step), 1.0))
        elif type(bin_step) is list:
            return pd.Series(np.append(np.arange(0.0, 1.0, bin_step[layer_idx][head_idx]), 1.0))
        else:
            return None

    if single_head_idx is None:
        # walk through layers and heads
        for layer_idx, layer in enumerate(data):
            print('plotting histogram for layer {}...'.format(layer_idx))
            atten_layers = {}
            for head_idx, head in enumerate(layer[0]):
                sparsity = (head <= (sparsity_bar)).sum() / head.flatten().shape[0]
                atten_layers['head_{}, max: {:.4f}, min: {:.4f}, spars: {:.4f}, sparsity_bar: {:.4f}'.format(
                    head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar)] = head.flatten().tolist()

            atten_layers_pd = pd.DataFrame(atten_layers)
            # create vars for plotting
            fig, ax = plt.subplots(3, 4, figsize=(21, 12))
            # extract pd column name and column into head_idx and head respectively
            for head_idx, head in atten_layers_pd.iteritems():
                head_idx_int = int(head_idx.split(',')[0].split('_')[1])
                curr_ax = ax[int(head_idx_int/4), int(head_idx_int % 4)]
                head.hist(ax=curr_ax, bins=get_bin_edges(bin_step, layer_idx, head_idx),
                          weights=(np.ones_like(head) / len(head)), color='C0')
                head.hist(ax=curr_ax, bins=get_bin_edges(bin_step, layer_idx, head_idx),
                          weights=(np.ones_like(head) / len(head)), cumulative=True,
                          histtype='step', linewidth=1, color='C3')
                curr_ax.set_xlim([0.0, 0.03])
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
        # compute cdf and draw as line
        # np_hist, np_bins = np.histogram(head, bins=get_bin_edges(bin_step, head))
        # np_hist_cumulative = np.insert(np.cumsum(np.multiply(np_hist, (np.ones_like(np_hist) / np.sum(np_hist)))), 0, 0.0)
        # np_bins = np.insert(np.add(np_bins[:-1], bin_step/2.0), 0, 0.0)
        # ax.plot(np_bins, np_hist_cumulative, 'k--', color='C3', linewidth=1)
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


def plot_heatmap(data, sparsity_bar=0.025, auto_scale=False, binarize=True, layer_aggregration='mean', attached_title=''):
    '''
    Plot the heat map to visualize the relation between each subwords in the
    self attention of each attention head in each layer

    expected data shape: (#layers, #heads, length, dv)
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


def plot_sparsity_change(data, sparsity_bar=0.025):
    spars_threshold = [float(i) for i in data.index.tolist()]
    for layer_idx in range(0, 12):
        print('plotting curve for sparsities...')
        fig, ax = plt.subplots(3, 4, figsize=(21, 12))
        for head_idx in range(0, 12):
            curr_ax = ax[int(head_idx/4), int(head_idx % 4)]
            curr_ax.plot(spars_threshold, data['layer_{}_head_{}'.format(
                layer_idx, head_idx)].tolist(), color='C0', marker='s')
            curr_ax.set_title('head {}'.format(head_idx))
            curr_ax.grid(linestyle='--', color='grey', alpha=0.6)
            curr_ax.set_ylim([0.0, 1.01])
            curr_ax.set_xlim(0.0, max(spars_threshold)+0.01)

        fig.suptitle('Sparsity for Different Thresholds with Fixed Bar for Layer {}'.format(
            layer_idx), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(RES_FIG_PATH+'spars_change_layer{}.png'.format(layer_idx), dpi=600)
        plt.clf()
        plt.close(fig)


if __name__ == '__main__':
    em_score, h_states, attens = get_hstates_attens(
        "csarron/roberta-base-squad-v1", filter_inputs=False, force_reinfer=False, single_input=False)
    em_str = 'EM={:.2f}'.format(em_score*100)
    # # plot histogram for all layers and all heads
    # plot_dist(attens, bin_step=0.0005, sparsity_bar=0.0005, attached_title=em_str)
    # # plot histogram for a certain head in a certain layer
    # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
    #           single_head_idx=(0, 0), attached_title=em_str)
    # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
    #           single_head_idx=(0, 9), attached_title=em_str)
    # plot_dist(attens, bin_step=200, sparsity_bar=0.0005,
    #           single_head_idx=(0, 11), attached_title=em_str)

    # # plot heatmaps
    # plot_heatmap(attens, sparsity_bar=0.0005, binarize=False, attached_title=em_str)
    # plot_heatmap(attens, sparsity_bar=0.0005, binarize=True, attached_title=em_str)
    # plot_heatmap(attens, sparsity_bar=0.0005, binarize=False,
    #              auto_scale=True, attached_title=em_str)
    # # compute sparsity
    # spars_threshold = ['0.0', '0.0005', '0.001', '0.005', '0.01', '0.05', '0.1']
    # spars = get_sparsities(spars_threshold, sparsity_bar=0.0005)
    # #print average sparsity change
    # print(spars.mean(axis=1))
    # plot_sparsity_change(spars, sparsity_bar=0.0005)
