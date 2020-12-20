"""
roberta base analyzer: analyzer sparsity of the roberta base 
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import transformer_visualization as tv

import argparse as ag
import os
import sys
import random
from datetime import datetime
import math
import glob
from textwrap import wrap
from itertools import compress, product
import json

PARAM_PATH = "./params/"
DATA_PATH = "./data"

def extract_inst_wikipedia(model_name, num_sentences: int):
    dataset = load_dataset("wikipedia", "20200501.en", cache_dir=DATA_PATH, split='train[:10%]')
    random.seed(12331)
    dataset = random.sample(dataset['text'], num_sentences)
    insts = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for doc in dataset:
        para = doc.split('\n\n')[0]
        tokenized_para_len = len(tokenizer(para)['input_ids'])       
        if(100 < tokenized_para_len < 512): insts.append(para)
   
    print("extracted {} paragrahps from wikipedia".format(len(insts)))
    return insts 


def extract_inst_squad(num_paras: int):
    data, squad_ver = [], 'v1.1'
    with open(DATA_PATH + '/dev-v1.1.json', "r", encoding="utf-8") as data_file:
        squad_raw_data = json.load(data_file)["data"]

        for topic in squad_raw_data:
            for pgraph in topic["paragraphs"]:
                data.append(pgraph["context"])

    random.seed(123)
    data = random.sample(data, num_paras)
    return data


def prepare_masked_tokens(model_name, num_sentences, device):
    
    input_tokens, labels = [], []
    tokens_path_list = [i.replace('\\', '/') for i in glob.glob(DATA_PATH + '/mlm_tokens_*.npz')]
    labels_path_list = [i.replace('\\', '/') for i in glob.glob(DATA_PATH + '/mlm_labels_*.npy')]

    if (len(tokens_path_list) > 0) and (len(labels_path_list) > 0):
        for token_f, label_f in zip(tokens_path_list, labels_path_list):
            token_f_res, temp_token = np.load(token_f), {}
            for key in token_f_res.files:
                temp_token[key] = token_f_res[key]
            input_tokens.append(temp_token)
            labels.append(np.load(label_f))

    else:
        sentences = extract_inst_wikipedia(model_name, num_sentences)
        # sentences = extract_inst_squad(num_sentences)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for inst_str in sentences:
            input_token = tokenizer([inst_str], padding=True, return_tensors='np')
            ids = input_token['input_ids']
            random_idx = np.random.choice(np.arange(start=1, stop=(ids.shape[-1]-1)), int(ids.shape[-1]*0.15), replace=False)
            masked_ids = np.stack([ids[0]]* len(random_idx))
            label = np.ones(masked_ids.shape) * -100
            for idx in range(masked_ids.shape[0]): 
                label[idx][random_idx[idx]] = masked_ids[idx][random_idx[idx]]
                masked_ids[idx][random_idx[idx]] = tokenizer.mask_token_id
            input_token['input_ids'] = masked_ids
            input_token['attention_mask'] = np.stack([input_token['attention_mask'][0]] * len(random_idx))
            if input_token.get('token_type_ids', None) is not None:
                input_token['token_type_ids'] = np.stack([input_token['token_type_ids'][0]] * len(random_idx))
            input_tokens.append(input_token)
            labels.append(label)

        for idx, i in enumerate(input_tokens):
            np.savez(DATA_PATH + '/mlm_tokens_{}.npz'.format(idx), **i)
        for idx, i in enumerate(labels):
            np.save(DATA_PATH + '/mlm_labels_{}.npy'.format(idx), i)

    for idx in range(len(input_tokens)):
        for k in input_tokens[idx].keys():
            input_tokens[idx][k] = torch.Tensor(input_tokens[idx][k]).to(device).long()
        labels[idx] = torch.Tensor(labels[idx]).to(device).long()

    return input_tokens, labels


# helper func: convert attention to numpy array in 
# list of [inst, [layers, heads, rows, cols]]
def convert_att_to_np(x, attn_mask):
    attn_mask = (torch.sum(attn_mask, dim=-1)).cpu().numpy()
    temp, res = np.asarray([layer.cpu().numpy() for layer in x]), []
    for i in range(temp.shape[1]):
        res.append(np.squeeze(temp[:, i, :, :attn_mask[i], :attn_mask[i]]))
    return res

def convert_hist_to_np(x): return np.asarray([layer.cpu().numpy() for layer in x])

def get_atten_hist_from_model(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0, stored_attentions=False, device='cuda'):
    param_file_path = PARAM_PATH + model_name
    head_mask = None

    attentions, attn_mask, hists = None, None, None
    if os.path.isfile(param_file_path + "_attention.npy") and stored_attentions:
        print("loading parameters from file...")
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file, allow_pickle=True)
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file) for i in range(len(attn_mask))]
        with open(param_file_path + "_hists.npy", "rb") as hists_file:
            hists = np.load(hists_file)
    else:
        sentences = extract_inst_wikipedia(model_name, num_sentences)
        # sentences = extract_inst_squad(num_sentences)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_tokens = tokenizer(sentences, padding=True, return_tensors='pt')
        for k in input_tokens.keys():
            input_tokens[k] = input_tokens[k].to(device)

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        if torch.cuda.is_available(): model = model.to(device)

        # run model
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True, \
                                    att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask)
        
        attentions = convert_att_to_np(model_output[2], input_tokens['attention_mask'])
        hists = convert_hist_to_np(model_output[1])

        if stored_attentions:
            with open(param_file_path + "_attention_mask.npy", "wb+") as att_mask_file:
                np.save(att_mask_file, attn_mask)
            with open(param_file_path + "_attention.npy", "wb+") as att_file:
                for i in range(len(attn_mask)): np.save(att_file, attentions[i], allow_pickle=False)
            with open(param_file_path + "_hists.npy", "wb+") as hists_file:
                np.save(hists_file, hists, allow_pickle=False)
        
    print ("Shape of attention weight matrices", len(attentions), attentions[0].shape)
    return attentions, hists

def evaluate_model(model_name: str, input_tokens, labels=None, att_threshold=0.0, hs_threshold=0.0, quant_base=0.0, device='cuda'):
    head_mask = None

    attentions, attn_mask, hists, loss = None, None, None, 0.0
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    if torch.cuda.is_available(): model = model.to(device)
    # run model
    with torch.no_grad():
        model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True, labels=labels, \
                                att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask, quantize=quant_base)
    
    
    attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
    
    hists = convert_hist_to_np(model_output[2])
    loss = (model_output[0]).item() * labels.size()[0]
        
    print ("Shape of attention weight matrices", len(attentions), attentions[0].shape)
    return loss, attentions, hists

def get_em_sparsity_from_masked_lm(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0, quant_base=0.0, device='cuda'):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    total_score, inst_count, all_max, all_min, all_mean, all_std, all_sparsity = \
        None, None, None, None, None, None, None
    
    # read from file
    input_type = "_all"
    score_path, att_stat_path = (PARAM_PATH + i + input_type + '.npy' \
                                    for i in ['score', 'att_stat_features'])
    if os.path.isfile(score_path) and os.path.isfile(att_stat_path):
        print("Loading parameters from file {}...".format(PARAM_PATH + input_type))
        with open(score_path, "rb") as score_file:
            total_score, inst_count = (i for i in np.load(score_file))
        with open(att_stat_path, "rb") as att_stat_file:
            all_max = np.load(att_stat_file)
            all_min = np.load(att_stat_file)
            all_mean = np.load(att_stat_file)
            all_std = np.load(att_stat_file)
            all_sparsity = np.load(att_stat_file)
    # extract parameters from model
    else:
        res, total_elem_count = None, 0
        inst_count = 0
        # fetch data
        all_input_tokens, all_labels = prepare_masked_tokens(model_name, num_sentences, device=device)

        for batch_inputs, batch_labels in zip(all_input_tokens, all_labels):
            inst_count += len(batch_labels)
            ppl, attentions, hidden_states = \
                evaluate_model(model_name, batch_inputs, batch_labels, att_threshold=att_threshold, hs_threshold=hs_threshold, device=device, quant_base=quant_base)
                
            def get_spars(x, axis): 
                return x.shape[-1] ** 2 - np.count_nonzero(x[:, :, :, :], axis=axis)
            def agg_func(f): return np.stack([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            def add_func(f): return np.sum([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            if res is None:
                res = {'score': ppl, 'mean': agg_func(np.mean),
                    'max': agg_func(np.amax), 'min': agg_func(np.amin), 
                    'std': agg_func(np.std), 'sparsity': add_func(get_spars)}
            else:
                res['score'] += ppl
                res['max'] = np.concatenate((res['max'], agg_func(np.amax)), axis=0)
                res['min'] = np.concatenate((res['min'], agg_func(np.amin)), axis=0)
                res['mean'] = np.concatenate((res['mean'], agg_func(np.mean)), axis=0)
                res['std'] = np.concatenate((res['std'], agg_func(np.std)), axis=0)
                res['sparsity'] = np.add(res['sparsity'], add_func(get_spars))

            total_elem_count += sum([att.shape[-1] * att.shape[-1] for att in attentions])

        res['sparsity'] = res['sparsity'].astype(float) / total_elem_count
        res['score'] /= float(inst_count)
        res['score'] = math.exp(res['score'])

        # save params
        total_score, all_max, all_min, all_mean, all_std, all_sparsity = \
            res['score'], res['max'], res['min'], res['mean'], res['std'], res['sparsity']

        with open(score_path, "wb+") as scores_file:
            np.save(scores_file, np.array([total_score, inst_count]))
        with open(att_stat_path, "wb+") as att_stat_file:
            np.save(att_stat_file, all_max)
            np.save(att_stat_file, all_min)
            np.save(att_stat_file, all_mean)
            np.save(att_stat_file, all_std)
            np.save(att_stat_file, all_sparsity)
    
    print("total score: ", total_score, "#instances: ", inst_count,
          "max dim:", all_max.shape, "min dim:", all_min.shape,
          "mean dim:", all_mean.shape, "std dim:", all_std.shape,
          "sparsity dim:", all_sparsity.shape)

    print(all_sparsity)

def get_sparse_hist_token(attn, offset, sparsity_bar=0.0):
    all_sparse_count = None
    for att in attn:
        curr_sparse_count = np.apply_along_axis(lambda a: float((a <= (sparsity_bar + offset)).sum()) / att.shape[-1], -1, att)
        all_sparse_count = curr_sparse_count if all_sparse_count is None \
                                else np.concatenate((curr_sparse_count, all_sparse_count), axis=-1)
    
    sparse_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=10, range=(0.0, 1.0))[0], -1, all_sparse_count)
    sparse_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, sparse_hist)
    return sparse_hist


def get_sampled_tokens(model_name, num_tokens, head_idx=(0, 0)):
    """
    getting sampled tokens from varies instances
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if torch.cuda.is_available(): model = model.to("cuda")
    
    # fetch data:
    insts = extract_inst_wikipedia(model_name, 20)
    input_tokens = tokenizer.batch_encode_plus(insts, padding=True, return_tensors="pt")   
    # run model
    if torch.cuda.is_available(): 
        for i in input_tokens.keys():
            input_tokens[i] = input_tokens[i].to("cuda")
        
    with torch.no_grad():
        model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True)
    attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
    
    
    all_tokens, all_attention_hists = [], None
    for i in input_tokens.keys():
        input_tokens[i] = input_tokens[i].to("cpu")

    for inst, attention in zip(input_tokens['input_ids'], attentions):
        attention = attention[head_idx[0], head_idx[1], :, :]
        all_tokens += [tokenizer.decode([i]).replace(' ', '') for i in inst[:attention.shape[-1]]]
        offset = 1e-8
        hist_x_start, hist_x_end = tv.log(offset, 10), tv.log(1, 10)
        attention_hist = np.apply_along_axis(
            lambda x: np.histogram(x, bins=tv.get_bin_edges(100, hist_x_start, hist_x_end, 'log'), range=(0.0, 1.0))[0], -1, attention)
        all_attention_hists = attention_hist if all_attention_hists is None \
                else np.concatenate((all_attention_hists, attention_hist), axis=0)
    
    sampled_tokens, sampled_attention_hists = [], None
    sparse_token_counter = 0
    for token, attention_hist in zip(all_tokens, all_attention_hists):
        norm_cdf = np.cumsum(attention_hist).astype("float") / np.sum(attention_hist)
        if norm_cdf[50] < 0.1:
            print(token)
            sampled_tokens.append(token)
            attention_hist = attention_hist.reshape((1, attention_hist.shape[0]))
            sampled_attention_hists = attention_hist if sampled_attention_hists is None \
                                        else np.concatenate((sampled_attention_hists, attention_hist), axis=0)
            sparse_token_counter += 1
        # if sparse_token_counter > 6:
        #     break
    
    random.seed(datetime.now())
    sampled_token_ids = random.sample(range(len(all_tokens)), num_tokens - sparse_token_counter)
    sampled_attention_hists = np.concatenate((sampled_attention_hists, all_attention_hists[sampled_token_ids, :]), axis=0)
    sampled_attention_hists = np.apply_along_axis(lambda a: a / np.sum(a), -1, sampled_attention_hists)
    return sampled_tokens, sampled_attention_hists, offset



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
    insts = extract_inst_wikipedia(model_name, num_sentences)
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
    model_name = 'bert-base-uncased'
    arg_parser = ag.ArgumentParser(description=__doc__)
    arg_parser.add_argument("-at", "--att_threshold", default=0.0,
                            required=False, help="set attention sparsity threshold")
    arg_parser.add_argument("-ht", "--hs_threshold", default=0.0,
                            required=False, help="set hidden states sparsity threshold")
    arg_parser.add_argument("-d", "--distribution", default=False, action='store_true',
                            required=False, help="print histogram")
    arg_parser.add_argument("-e", "--evaluation", default=False, action="store_true",
                            required=False, help="evaluate model only without any plot")
    arg_parser.add_argument("-sa", "--samples", default=-1,
                            required=False, help="number of samples for distribution")
    arg_parser.add_argument("-qb", "--quantize_base", default=0.0,
                            required=False, help="base for quantization")

    args = vars(arg_parser.parse_args())
    att_threshold = float(args['att_threshold'])
    hs_threshold = float(args['hs_threshold'])
    samples = int(args['samples'])
    quant_base = float(args['quantize_base'])

    # list_sparse_tokens_all("roberta-base", sparsity_bar=1e-8, num_sentences=8000)
    if args['evaluation']:
        get_em_sparsity_from_masked_lm(model_name, samples, att_threshold=att_threshold, hs_threshold=hs_threshold, quant_base=quant_base)

    if args['distribution']:
        attns, hists = get_atten_hist_from_model(model_name, samples, att_threshold=att_threshold, hs_threshold=hs_threshold)
        attn_mask = [i.shape[-1] for i in attns]
        print(hists.shape, len(attn_mask))

        # h_state sanity check
        for i in range(10):
            print("h_state mean:{:.4f}, std:{:.4f}".format(
                np.mean(hists[0][0][i*5], axis=-1), np.std(hists[0][0][i*5], axis=-1)))
        tv.plot_atten_dist_per_token(attns, 100, sparse_hist=get_sparse_hist_token(attns, 1e-8))
        tv.plot_atten_dist_per_token_compare(attns, 100, [(0, 2), (0, 5)])
        tv.plot_hs_dist_per_token(hists, 100, attn_mask, scale='linear')
