"""
roberta base analyzer: analyzer sparsity of the roberta base 
"""

from transformers import pipeline
from transformers import BertForMaskedLM, BertTokenizer
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
    dataset = load_dataset("wikipedia", "20200501.en", cache_dir=DATA_PATH, split='train[:10%]')
    random.seed(12331)
    dataset = random.sample(dataset['text'], num_sentences)
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

def run_mask_fill_pipeline(model_name: str, num_sentences):
    mf_pipeline = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=0, \
        topk=20, att_threshold=0.0001, output_attentions=True, output_hidden_states=True)

    # fetch data:
    insts, masked_insts, correct_token = extract_inst_wikipedia(num_sentences), [], []
    for inst in insts:
        word_list = inst.split(' ')
        rand_token_to_mask = random.randint(0, len(word_list)-1)
        correct_token.append(word_list[rand_token_to_mask])
        word_list[rand_token_to_mask] = '[MASK]'
        masked_insts.append(" ".join(word_list))

    results = mf_pipeline(masked_insts)
    
    correct = 0
    for res_idx, res in enumerate(results['result']):
        for possible_res in res:
            if possible_res['token_str'].lower() == correct_token[res_idx].lower():
                correct += 1
                break

    print("em: ", float(correct)/num_sentences)

    return float(correct)/num_sentences, results['hidden_states'], results['attentions']
    

def get_atten_hist_from_model(model_name: str, num_sentences: int):
    param_file_path = PARAM_PATH + model_name

    attentions, attn_mask, hists, sparse_hist = None, None, None, None
    if os.path.isfile(param_file_path + "_attention.npy"):
        print("loading parameters from file...")
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file)
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file) for i in range(len(attn_mask))]
        with open(param_file_path + "_hists.npy", "rb") as hists_file:
            hists = np.load(hists_file)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        if torch.cuda.is_available(): model = model.to("cuda")
        
        # fetch data:
        insts, masked_insts = extract_inst_wikipedia(num_sentences), []
        for inst in insts:
            word_list = inst.split(' ')
            word_list[random.randint(0, len(word_list)-1)] = '[MASK]'
            masked_insts.append(" ".join(word_list))

        input_tokens = tokenizer(masked_insts, padding=True, return_tensors="pt")
        labels = tokenizer(insts, padding=True, return_tensors="pt")['input_ids']

        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
            labels = labels.to("cuda")
              
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True, labels=labels)
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


def get_sparse_hist_token(attn, offset, sparsity_bar=0.0):
    all_sparse_count = None
    for att in attn:
        curr_sparse_count = np.apply_along_axis(lambda a: float((a <= (sparsity_bar + offset)).sum()) / att.shape[-1], -1, att)
        all_sparse_count = curr_sparse_count if all_sparse_count is None \
                                else np.concatenate((curr_sparse_count, all_sparse_count), axis=-1)
    
    sparse_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=10, range=(0.0, 1.0))[0], -1, all_sparse_count)
    sparse_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, sparse_hist)
    return sparse_hist


def attn_head_row_count(attn): return attn.shape[-1] * attn.shape[1]

def attn_token_layer_count(attn): return attn_head_row_count(attn) * attn.shape[0]

def get_sparse_token(attn, sparsity_bar, return_count=False):
    """
    compute the most sparse token per head, per layer. The input is the attention for one instance 
    with shape (#layer, #head, length, length)
    """        
    sparse_per_row = np.count_nonzero(attn <= sparsity_bar, axis=(1, -1))
    sparse_per_row_all = \
        sparse_per_row.transpose((1, 0))
    sparse_per_row_all = np.sum(sparse_per_row_all, axis=-1)
    if not return_count:
        sparse_per_row = sparse_per_row / attn_head_row_count(attn)
        sparse_per_row_all = sparse_per_row_all / attn_token_layer_count(attn)
    
    return sparse_per_row, sparse_per_row_all

def list_sparse_tokens_per_inst(model_name, sparsity_bar=0.0, num_sentences=1):
    attentions, hists = None, None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available(): model = model.to("cuda")
    
    # fetch data:
    insts = extract_inst_wikipedia(num_sentences)
    # insts = ["The girl ran to a local pub to escape the din of her city."]

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

def list_sparse_tokens_all(model_name, sparsity_bar=0.0, num_sentences=500):
    attentions = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available(): model = model.to("cuda")
    
    # fetch data:
    insts = extract_inst_wikipedia(num_sentences)
    # insts = ["The girl ran to a local pub to escape the din of her city.", 
    #         "I am a robot writing fantastic articles just like human-being.", 
    #         "Today is a beautiful day at new england area."]

    sparse_count_list = {}
    for inst_idx, inst in enumerate(insts):
        input_tokens = tokenizer.encode_plus(inst, add_special_tokens=True, return_tensors="pt")
        if input_tokens['input_ids'].size()[-1] > 512:
            continue
        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
            
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True)

        attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])

        for tokens, attn in zip(input_tokens['input_ids'], attentions):
            _, sparsity_all = get_sparse_token(attn, sparsity_bar, return_count=True)
            for token, sparse_count in zip(tokens, sparsity_all):
                token_str = tokenizer.decode([token]).replace(' ', '')
                sparse_count_list[token_str] = (sparse_count_list.get(token_str, (0, 0))[0] + sparse_count, 
                                            sparse_count_list.get(token_str, (0, 0))[1] + attn_token_layer_count(attn))


    with open("token_sparse_list_all.txt", "w+", newline="") as f:
        token_sparse_list = pd.DataFrame({'tokens': sparse_count_list.keys(),
                                            'sparse_count': [i[0] for i in list(sparse_count_list.values())],
                                            'all_count': [i[1] for i in list(sparse_count_list.values())],
                                            'sparsity_all': [float(i[0])/float(i[1]) for i in list(sparse_count_list.values())]})
        listed_tokens = token_sparse_list[token_sparse_list['all_count'] > token_sparse_list['all_count'].nlargest(71).iloc[-1]]
        f.write(listed_tokens.sort_values('sparsity_all', ascending=False).to_string())


if __name__ == "__main__":
    # list_sparse_tokens_all("roberta-base", sparsity_bar=1e-8, num_sentences=8000)

    run_mask_fill_pipeline('bert-base-uncased', 5)
    exit()

    attns, hists = get_atten_hist_from_model('bert-base-uncased', 3)
    attn_mask = [i.shape[-1] for i in attns]
    print(hists.shape, len(attn_mask))

    # h_state sanity check
    for i in range(10):
        print("h_state mean:{:.4f}, std:{:.4f}".format(
            np.mean(hists[0][0][i*5], axis=-1), np.std(hists[0][0][i*5], axis=-1)))
    # tv.plot_atten_dist_per_token(attns, 100, sparse_hist=get_sparse_hist_token(attns, 0.0))
    # tv.plot_hs_dist_per_token(hists, 100, attn_mask, scale='linear')
