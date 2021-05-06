"""
squad related task quantization tool
"""

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
import torch
import numpy as np
import pandas as pd
from scipy.special import kl_div
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
from tqdm import tqdm
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

def linear_quant_clamped(att, bits, min_val, max_val):
    base = (max_val - min_val) / (2**int(bits)-1)
    cutpoints = [0.0] + [(i+1)*base for i in range(int(2.0**bits-1))]
    res = np.floor((att - min_val) / base) * base + min_val
    res[att < (cutpoints[1]+min_val)] = 0.0 
    res[att > max_val] = max_val
    return res

def quant_8_log(att, bits, min_val, max_val):
    min_exp, max_exp = log2(min_val), log2(max_val)
    base = (max_exp-min_exp) / (2.0**bits - 1)
    cutpoints = [0.0] + [(i+1)*base for i in range(int(2.0**bits-1))]
    print(cutpoints)
    res = np.floor((np.log2(att)-min_exp) / base) * base + min_exp
    res[att < 2**(cutpoints[1]+min_exp)] = float('-Inf')
    res[att > max_val] = max_exp
    print(res)
    return 2**res

def search_max_min(att, bits, method='linear'):
    methods = {'linear': linear_quant_clamped, 'log': quant_8_log}
    all_att = np.concatenate([i.flatten() for i in att], axis=0)
    original_histogram = np.histogram(all_att, bins=200, range=(0.0, 1.0), weights=np.full(all_att.shape, 1./all_att.shape[0]))
    
    curr_min_kl, curr_min_val, curr_max_val = float('inf'), 0.0, 1.0
    for min_val, max_val in tqdm(list(product(np.arange(0.0, 0.002, 0.0001), np.arange(0.99, 1, 0.001)))):
        quantized_att = methods[method](all_att, bits, min_val, max_val)
        quantized_histogram = np.histogram(quantized_att, bins=200, range=(0.0, 1.0), weights=np.full(quantized_att.shape, 1./quantized_att.shape[0]))
        kl = kl_div(original_histogram[0], quantized_histogram[0])
        kl = np.mean(kl[kl < float('inf')])
        if kl < curr_min_kl:
            curr_min_kl = kl
            curr_max_val = max_val
            curr_min_val = min_val

    print(f'minimum min_val and kl divergence: {curr_min_val}, {curr_max_val}, {kl}')


def search_max_min_original(att, bits, num_bins = 2048):
    all_att = np.concatenate([i.flatten() for i in att], axis=0)
    original_histogram, edges = np.histogram(all_att, bins=num_bins, range=(0.0, 1.0))
    hist_width = 1./2048.

    min_kl = float('inf')
    idx_res = 0
    for i in tqdm(list(np.arange(int(2**bits)+1, num_bins))):
        ref_dist_p = np.array(original_histogram[num_bins-i:])
        outliers_count = np.sum(original_histogram[:num_bins-i])
        ref_dist_p[0] += outliers_count

        ref_dist_q_bins = np.array_split(ref_dist_p, int(2**bits))
        ref_dist_q = []
        for j in ref_dist_q_bins:
            single_bin = [np.sum(j) / np.sum(j > 0)] * j.shape[0] * (j > 0)
            ref_dist_q.extend(single_bin)

        ref_dist_p = ref_dist_p / np.sum(ref_dist_p)
        ref_dist_q = ref_dist_q / np.sum(ref_dist_q)
        kl = kl_div(ref_dist_p, ref_dist_q)
        kl = np.mean(kl[kl < float('inf')])
        if kl < min_kl:
            min_kl = kl
            idx_res = num_bins - i

    print(f'minimum min_val and kl divergence: {(idx_res+0.5)*hist_width}, {min_kl}')

if __name__ == '__main__':
    # open dumped attention
    all_attentions = None

    input_type = "_sampled"
    atten_path = PARAM_PATH + 'attentions_sampled.npy'
    if os.path.isfile(atten_path):
        print("Loading parameters from file {}...".format(PARAM_PATH + input_type))
        atten_len = 0
        with open(atten_path, "rb") as attention_file:
            atten_len, all_attentions = (np.load(attention_file))[0], []
            for i in range(atten_len): all_attentions.append(np.load(attention_file))

    print(f'{atten_len} instances has been loaded.')

    search_max_min_original(all_attentions, 7.0)
