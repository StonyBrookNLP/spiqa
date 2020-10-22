"""
roberta base analyzer: analyzer sparsity of the roberta base 
"""

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
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
from textwrap import wrap
from itertools import compress, product

RES_FIG_PATH = "./res_fig/"
PARAM_PATH = "./params/"
DATA_PATH = "./data/"
FILT_PARAM_PATH = "./filtered_params/"

def extract_inst_wikipedia(num_sentences: int):
    dataset = load_dataset("wikipedia", "20200501.en", cache_dir=DATA_PATH)
    # dataset = random.sample(dataset['train']['text'], num_sentences)
    dataset = dataset[:num_sentences]
    return dataset

def get_attention_from_model(model_name: str, num_sentences: int):
    # helper func: convert attention to numpy array in 
    # list of [inst, [layers, heads, rows, cols]]
    def convert_att_to_np(x, attn_mask): 
        temp, res = np.asarray([layer.cpu().numpy() for layer in x]), []
        for i in range(temp.shape[1]):
            res.append(np.squeeze(temp[:, i, :, :attn_mask[i], :attn_mask[i]]))
        return res
    
    param_file_path = PARAM_PATH + "_" + model_name

    attentions, attn_mask = None, None
    if os.path.isfile(param_file_path + "_attention.npy"):
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file)
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file) for i in range(len(attn_mask))]
    else:
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_config(config)
        if torch.cuda.is_available(): model = model.to("cuda")
        
        # fetch data:
        insts = extract_inst_wikipedia(num_sentences)
        input_tokens = tokenizer.batch_encode_plus(insts, padding=True, return_tensors="pt")

        print(input_tokens[:3])
        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
                
        model_output = model(**input_tokens)
        attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
        attn_mask = input_tokens['attention_mask'].cpu().numpy()
        with open(param_file_path + "_attention_mask.npy", "wb+") as att_mask_file:
            np.save(att_mask_file, attn_mask, allow_pickle=False)
        with open(param_file_path + "_attention.npy", "wb+") as att_file:
            for i in range(len(attn_mask)): np.save(att_file, attentions[i], allow_pickle=False)
        
    print ("Shape of attention weight matrices", len(attentions), attentions.shape)

    return attentions

if __name__ == "__main__":
    get_attention_from_model('roberta-base', 10)