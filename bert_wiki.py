import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
#You can also use specific model classes which would be a subclass to the abovementioned "Auto" classes.
#from transformers import RobertaModel, RobertaTokenizer

from pprint import pprint
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

import argparse as ag
import os
import sys
import random
import functools
from subprocess import call
from math import isnan, fsum, log

from roberta_squad_analyzer import plot_dist
import transformer_visualization as tv

offset = 1e-10
scale = "log"
hist_x_start, hist_x_end = log(offset, 10), log(1+offset, 10)
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

model_name = "bert-base-uncased" 
#config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if torch.cuda.is_available(): model = model.to("cuda")

wiki = load_dataset("wikipedia", "20200501.en", split='train')
sentences = []

#Taking only 2 instances from the Wiki dataset
for i in range(0, 2):
    sentences.append(wiki[i]['text'])
    
input_tokens = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors="pt")
#pprint (input_tokens)

if torch.cuda.is_available(): 
    for i in input_tokens.keys():
        input_tokens[i] = input_tokens[i].to("cuda")
        
model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True)

#Retrieving attentions for all layers for all instances
layer_num = 12
all_attens = []
for i in range(input_tokens["attention_mask"].shape[0]):
    num_tokens = torch.sum(input_tokens["attention_mask"][i]).item()
    total = []
    for j in range(0, layer_num):
        sentence_rep = model_output[3][j][i, :, :, :num_tokens]
        total.append(sentence_rep)
    a = torch.stack(total, 0)
    all_attens.append(a)

#Plotting distributions for each token per layer and head
all_hist, all_max, all_min, all_seq_len, all_sparse_count = None, None, None, None, None

atten_bins, atten_hist = get_bin_edges(100), None
sparsity_bar = 0.0
for att in all_attens:
    att = att[:,:,:att.shape[-1],:]
    att = att.cpu().detach().numpy()
    #print("min:", torch.min(att[att.nonzero()]))
    curr_hist = np.apply_along_axis(lambda a: np.histogram(a+offset, atten_bins)[0], -1, att)
    atten_hist = [curr_hist] if atten_hist is None else atten_hist + [curr_hist]
    curr_sparse_count = np.apply_along_axis(lambda a: float((a <= sparsity_bar).sum()) / att.shape[-1], -1, att)
    all_sparse_count = curr_sparse_count if all_sparse_count is None \
                        else np.concatenate((curr_sparse_count, all_sparse_count), axis=-1)
    all_seq_len = [att.shape[-1]] if all_seq_len is None else all_seq_len + [att.shape[-1]]
    curr_max, curr_min = np.amax(att, axis=(-2, -1)), np.amin(att, axis=(-2, -1))

all_max = curr_max if all_max is None else np.maximum(all_max, curr_max)
all_min = curr_min if all_min is None else np.minimum(all_min, curr_min)

atten_hist = np.concatenate(atten_hist, axis=-2)
sparse_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=10, range=(0.0, 1.0))[0], -1, all_sparse_count)

atten_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, atten_hist)
sparse_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, sparse_hist)

tv.plot_atten_dist_per_token(atten_hist, 100, all_max, all_min, sparse_hist=sparse_hist)

# Plot cdf and pdf for each layer and each head using attentions from all tokens
# for a in all_attens:
#     plot_dist(a, bin_step=150, sparsity_bar=0.0005, layer_aggregration='None', attached_title="em_str")