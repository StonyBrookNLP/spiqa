"""
roberta base analyzer: analyzer sparsity of the roberta base 
"""

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import transformer_visualization as tv

import argparse as ag
import os
import sys
import random
from textwrap import wrap
from itertools import compress, product

PARAM_PATH = "./params/"
DATA_PATH = "./data"

def extract_inst_wikipedia(num_sentences: int):
    dataset = load_dataset("wikipedia", "20200501.en", cache_dir=DATA_PATH)
    random.seed(12331)
    dataset = random.sample(dataset['train']['text'], num_sentences)
    insts = []
    for doc in dataset:
        insts.append(doc.split('\n\n')[0])
    return insts 

# helper func: convert attention to numpy array in 
# list of [inst, [layers, heads, rows, cols]]
def convert_att_to_np(x, attn_mask):
    attn_mask = (torch.sum(attn_mask, dim=-1)).cpu().numpy()
    temp, res = np.asarray([layer.cpu().numpy() for layer in x]), []
    for i in range(temp.shape[1]):
        res.append(np.squeeze(temp[:, i, :, :attn_mask[i], :attn_mask[i]]))
    return res

def convert_hist_to_np(x): return np.asarray([layer.cpu().numpy() for layer in x])

def get_atten_hist_from_model(model_name: str, num_sentences: int):
    param_file_path = PARAM_PATH + model_name

    attentions, attn_mask, hists = None, None, None
    if os.path.isfile(param_file_path + "_attention.npy"):
        print("loading parameters from file...")
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file)
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file) for i in range(len(attn_mask))]
        with open(param_file_path + "_hists.npy", "rb") as hists_file:
            hists = np.load(hists_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available(): model = model.to("cuda")
        
        # fetch data:
        insts = extract_inst_wikipedia(num_sentences)
        input_tokens = tokenizer.batch_encode_plus(insts, padding=True, return_tensors="pt")

        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
              
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True)
        attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
        attn_mask = input_tokens['attention_mask'].cpu().numpy()
        hists = convert_hist_to_np(model_output[2])

        with open(param_file_path + "_attention_mask.npy", "wb+") as att_mask_file:
            np.save(att_mask_file, attn_mask, allow_pickle=False)
        with open(param_file_path + "_attention.npy", "wb+") as att_file:
            for i in range(len(attn_mask)): np.save(att_file, attentions[i], allow_pickle=False)
        with open(param_file_path + "_hists.npy", "wb+") as hists_file:
            np.save(hists_file, hists, allow_pickle=False)
        
    print ("Shape of attention weight matrices", len(attentions), attentions[0].shape)
    return attentions, hists

def get_sparse_token(attn, sparsity_bar):
    """
    compute the most sparse token per head, per layer. The input is the attention for one instance 
    with shape (#layer, #head, length, length)
    """        
    sparse_per_row = np.count_nonzero(attn <= sparsity_bar, axis=(1, -1))
    sparse_per_row_all = \
        sparse_per_row.transpose((1, 0))
    sparse_per_row_all = np.sum(sparse_per_row_all, axis=-1)
    sparse_per_row = sparse_per_row / (attn.shape[-1] * attn.shape[1])
    sparse_per_row_all = sparse_per_row_all / (attn.shape[-1] * attn.shape[0] * attn.shape[1])
    return sparse_per_row, sparse_per_row_all

def list_sparse_tokens(model_name, sparsity_bar=0.0, num_sentences=1):
    attentions, hists = None, None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available(): model = model.to("cuda")
    
    # fetch data:
    # insts = extract_inst_wikipedia(num_sentences)
    insts = ["The girl ran to a local pub to escape the din of her city."]

    for inst_idx, inst in enumerate(insts):
        input_tokens = tokenizer.encode_plus(inst, return_tensors="pt")
        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
            
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True)

        attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
        hists = convert_hist_to_np(model_output[2])

        sparse_table_path = "token_spars_list{}.txt".format(inst_idx)
        with open(sparse_table_path, 'w+') as f:
            for idx, attn in enumerate(attentions):
                tokens = [tokenizer.decode([i]) for i in input_tokens['input_ids'][idx]]
                sparsity_per_layer, sparsity_all = get_sparse_token(attn, sparsity_bar)
                spars_list = pd.DataFrame({'tokens': tokens, 'sparsity_all': sparsity_all})
                for layer in range(sparsity_per_layer.shape[0]):
                    spars_list['layer_{}'.format(layer)] = sparsity_per_layer[layer]
                f.write(spars_list.sort_values('sparsity_all', ascending=False).to_string())
            
            f.write("\n-------------\n")
            f.write(inst)


if __name__ == "__main__":
    list_sparse_tokens("roberta-base", sparsity_bar=0.0, num_sentences=5)
    list_sparse_tokens("roberta-base", sparsity_bar=0.0005, num_sentences=5)
    attns, hists = get_atten_hist_from_model('roberta-base', 10)
    attn_mask = [i.shape[-1] for i in attns]
    print(hists.shape, len(attn_mask))

    # h_state sanity check
    for i in range(10):
        print("h_state mean:{:.4f}, std:{:.4f}".format(
            np.mean(hists[0][0][i*5], axis=-1), np.std(hists[0][0][i*5], axis=-1)))
    tv.plot_atten_dist_per_token(attns, 100)
    tv.plot_hs_dist_per_token(hists, 100, attn_mask, scale='linear')
