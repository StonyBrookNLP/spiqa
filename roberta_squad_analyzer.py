from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import argparse
import os.path
from math import isnan
from textwrap import wrap

RES_FIG_PATH = "./res_fig/"
PARAM_PATH = "./params/"


def run_qa_pipeline():
    qa_pipeline = pipeline(
        "question-answering",
        model="csarron/roberta-base-squad-v1",
        tokenizer="csarron/roberta-base-squad-v1"
    )

    predictions = qa_pipeline({
        'context': "The game was played on February 7, 2016 at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
        'question': "What day was the game played on?"
    })

    print(predictions)


def get_hstates_attens(model_name: str):
    all_hidden_states, all_attentions = None, None
    # read from file
    if os.path.isfile(PARAM_PATH+'hidden_states.npy') and os.path.isfile(PARAM_PATH+'attentions.npy'):
        print("Loading parameters from file...")
        with open(PARAM_PATH+"hidden_states.npy", "rb") as h_states_file:
            all_hidden_states = np.load(h_states_file)
        with open(PARAM_PATH+"attentions.npy", "rb") as attention_file:
            all_attentions = np.load(attention_file)

        return all_hidden_states, all_attentions
    # extract parameters from model
    else:
        print("Parameter files not found, extracting them from model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        answers = ["HuggingFace is based in NYC",
                   "New York City is not the capital of New York State"]
        questions = ["Where is HuggingFace based?", "Is New York City a capital city?"]

        # encode and padding
        input_ids = []
        max_len = 0
        for qa_pair in zip(questions, answers):
            input_ids.append(tokenizer(qa_pair[0], qa_pair[1])["input_ids"])
            if len(input_ids[-1]) > max_len:
                max_len = len(input_ids[-1])

        padded_qa = torch.tensor([i + [0]*(max_len-len(i)) for i in input_ids])
        attention_mask = torch.tensor(np.where(padded_qa != 0, 1, 0))

        with torch.no_grad():
            outputs = model(padded_qa, attention_mask=attention_mask,
                            output_attentions=True, output_hidden_states=True, return_dict=True)

        def convert_var_to_np(x): return np.asarray(
            [layer.numpy() for layer in outputs[x]])
        all_hidden_states, all_attentions = \
            convert_var_to_np('hidden_states'), convert_var_to_np('attentions')
        print("hidden_state dim: ", all_hidden_states.shape,
              "attention dim:", all_attentions.shape)
        with open(PARAM_PATH+"hidden_states.npy", "wb+") as h_states_file:
            np.save(h_states_file, all_hidden_states)
        with open(PARAM_PATH+"attentions.npy", "wb+") as attention_file:
            np.save(attention_file, all_attentions)

    return all_hidden_states, all_attentions


def plot_dist(data, sparsity_bar=0.025, bin_step=float('Nan')):
    '''
    Plot the histrogram to visualize the distribution of the self attention 
    matrix for each attention head in each layer.

    expected data shape: (#layers, #heads, length, dv)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    '''
    print('plotting histrogram...')
    # walk through layers and heads
    for layer_idx, layer in enumerate(data):
        atten_layers = {}
        for head_idx, head in enumerate(layer):
            sparsity = (head <= (sparsity_bar)).sum() / head.flatten().shape[0]
            atten_layers['head_{}, max: {:.3f}, min: {:.3f}, spars: {:.3f}, sparsity_bar: {:.3f}'.format(
                head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar)] = head.flatten().tolist()

        atten_layers_pd = pd.DataFrame(atten_layers)
        # create vars for plotting
        fig, ax = plt.subplots(3, 4, figsize=(21, 12))
        # extract pd column name and column into head_idx and head respectively
        for head_idx, head in atten_layers_pd.iteritems():
            head_idx_int = int(head_idx.split(',')[0].split('_')[1])
            bin_edges = \
                50 if isnan(bin_step) \
                else pd.Series(np.append(np.arange(0.0, np.amax(head), bin_step), np.amax(head)))
            head.hist(ax=ax[int(head_idx_int/4), int(head_idx_int % 4)],
                      bins=bin_edges, weights=(np.ones_like(head) / len(head)))
            ax[int(head_idx_int/4), int(head_idx_int % 4)] \
                .set_title('\n'.join(wrap(head_idx, 34)))

        for axis in ax.flatten():
            axis.grid(linestyle='--', color='grey', alpha=0.6)
        fig.suptitle(
            'Histogram of Layer {}\'s Attention per head (batch aggregation=sum)'.format(layer_idx), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(RES_FIG_PATH+'hist_layer{}.svg'.format(layer_idx))
        plt.clf()


def plot_heatmap(data, sparsity_bar=0.025):
    '''
    Plot the heat map to visualize the relation between each subwords in the
    self attention of each attention head in each layer

    expected data shape: (#layers, #heads, length, dv)
    layers: layer_<0-11>
    sparsity_bar: threshold for sparsity calculation
    '''
    for layer_idx, layer in enumerate(data):
        fig, axs = plt.subplots(3, 4, figsize=(19, 12))
        for head_idx, head in enumerate(layer):
            sparsity = (head <= sparsity_bar).sum() / head.flatten().shape[0]
            ax = axs[int(head_idx/4), int(head_idx % 4)]
            c = ax.pcolor(head)
            fig.colorbar(c, ax=ax)
            info = 'head_{}, max: {:.3f}, min: {:.3f}, spars: {:.3f}, sparsity_bar: {:.3f}'.format(
                head_idx, np.amax(head), np.amin(head), sparsity, sparsity_bar)
            ax.set_title('\n'.join(wrap(info, 34)))

        fig.suptitle(
            'Heatmap of Layer {}\'s Attention per head (batch aggregation=sum)'.format(layer_idx), fontsize=21, y=0.99)
        fig.tight_layout()
        plt.savefig(RES_FIG_PATH+'heatmap_layer{}.svg'.format(layer_idx))
        plt.clf()


if __name__ == '__main__':
    h_states, attens = get_hstates_attens("csarron/roberta-base-squad-v1")
    # aggregrate attention by sum
    atten = np.sum(attens, axis=1)
    # plot_dist(atten, sparsity_bar=0.05, bin_step=0.05)
    plot_heatmap(atten, sparsity_bar=0.05)
