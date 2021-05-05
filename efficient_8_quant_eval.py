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

def search_max_min(att, bits):
    all_att = np.concatenate([i.flatten() for i in att], axis=0)
    original_histogram = np.histogram(all_att, bins=200, range=(0.0, 1.0), weights=np.full(all_att.shape, 1./all_att.shape[0]))
    
    curr_min_kl, curr_min_val, curr_max_val = float('inf'), 0.0, 1.0
    for min_val, max_val in tqdm(list(product(np.arange(0.0, 0.002, 0.0001), np.arange(0.99, 1, 0.001)))):
        quantized_att = linear_quant_clamped(all_att, bits, min_val, max_val)
        quantized_histogram = np.histogram(quantized_att, bins=200, range=(0.0, 1.0), weights=np.full(quantized_att.shape, 1./quantized_att.shape[0]))
        kl = kl_div(original_histogram[0], quantized_histogram[0])
        kl = np.mean(kl[kl < float('inf')])
        if kl < curr_min_kl:
            curr_min_kl = kl
            curr_max_val = max_val
            curr_min_val = min_val

    print(f'minimum min_val and kl divergence: {curr_min_val}, {curr_max_val}, {kl}')

if __name__ == '__main__':
    # open dumped attention
    all_attentions = None

    input_type = "_sampled"
    atten_path = PARAM_PATH + 'attentions' + input_type + '.npy'
    if os.path.isfile(atten_path):
        print("Loading parameters from file {}...".format(PARAM_PATH + input_type))
        atten_len = 0
        with open(atten_path, "rb") as attention_file:
            atten_len, all_attentions = (np.load(attention_file))[0], []
            for i in range(atten_len): all_attentions.append(np.load(attention_file))

    print(f'{atten_len} instances has been loaded.')

    search_max_min(all_attentions, 4.0)
