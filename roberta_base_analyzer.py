"""
roberta base analyzer: analyzer sparsity of the roberta base 
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import transformer_visualization as tv

import argparse as ag
import os
import sys
import random
import math
from textwrap import wrap
from itertools import compress, product

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
        if(tokenized_para_len < 512): insts.append(para)
   
    print("extracted {} paragrahps from wikipedia".format(len(insts)))
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
    

def get_atten_hist_from_model(model_name: str, insts, att_threshold=0.0, hs_threshold=0.0, stored_attentions=False):
    param_file_path = PARAM_PATH + model_name
    head_mask = None

    attentions, attn_mask, hists, loss = None, None, None, 0.0
    if os.path.isfile(param_file_path + "_attention.npy") and stored_attentions:
        print("loading parameters from file...")
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file, allow_pickle=True)
            loss = np.load(att_mask_file, allow_pickle=True)[0]
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file) for i in range(len(attn_mask))]
        with open(param_file_path + "_hists.npy", "rb") as hists_file:
            hists = np.load(hists_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        if torch.cuda.is_available(): model = model.to("cuda")

        input_tokens = tokenizer(insts, padding=True, return_tensors="pt")
        labels = torch.Tensor(np.ones(input_tokens['input_ids'].shape) * -100).type(torch.long)
        masked_token_idx = [random.randint(1, len(i)-2) for i in input_tokens['input_ids']]

        for inst_id, masked_idx in enumerate(masked_token_idx):
            labels[inst_id][masked_idx] = input_tokens['input_ids'][inst_id][masked_idx]
            input_tokens['input_ids'][inst_id][masked_idx] = tokenizer.mask_token_id

        # run model
        if torch.cuda.is_available(): 
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to("cuda")
                labels = labels.to("cuda")
              
        with torch.no_grad():
            model_output = model(**input_tokens, output_hidden_states=True, output_attentions=True, labels=labels, \
                                    att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask)
        
        
        attentions = convert_att_to_np(model_output[3], input_tokens['attention_mask'])
        attn_mask = input_tokens['attention_mask'].cpu().numpy()
        hists = convert_hist_to_np(model_output[2])
        loss = (model_output[0]).item()

        if stored_attentions:
            with open(param_file_path + "_attention_mask.npy", "wb+") as att_mask_file:
                np.save(att_mask_file, attn_mask)
                np.save(att_mask_file, np.array([loss]))
            with open(param_file_path + "_attention.npy", "wb+") as att_file:
                for i in range(len(attn_mask)): np.save(att_file, attentions[i], allow_pickle=False)
            with open(param_file_path + "_hists.npy", "wb+") as hists_file:
                np.save(hists_file, hists, allow_pickle=False)
        
    print ("Shape of attention weight matrices", len(attentions), attentions[0].shape)
    return loss, attentions, hists

def get_em_sparsity_from_masked_lm(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0):
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
        batch_size = 20
        # fetch data
        all_insts = list(chunks(extract_inst_wikipedia(model_name, num_sentences), batch_size))
        random.seed(6)

        for batch_inst in all_insts:
            ppl, attentions, hidden_states = \
                get_atten_hist_from_model(model_name, batch_inst, att_threshold=att_threshold, hs_threshold=hs_threshold)
                
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
        res['score'] /= float(len(all_insts))
        res['score'] = math.exp(res['score'])

        # save params
        total_score, inst_count, all_max, all_min, all_mean, all_std, all_sparsity = \
            res['score'], num_sentences, res['max'], res['min'], res['mean'], res['std'], res['sparsity']

        with open(score_path, "wb+") as scores_file:
            np.save(scores_file, np.array([total_score, inst_count]))
        with open(att_stat_path, "wb+") as att_stat_file:
            np.save(att_stat_file, all_max)
            np.save(att_stat_file, all_min)
            np.save(att_stat_file, all_mean)
            np.save(att_stat_file, all_std)
            np.save(att_stat_file, all_sparsity)
    
    print("total score: ", total_score, "#sentences: ", inst_count,
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

    args = vars(arg_parser.parse_args())
    att_threshold = float(args['att_threshold'])
    hs_threshold = float(args['hs_threshold'])
    samples = int(args['samples'])

    # list_sparse_tokens_all("roberta-base", sparsity_bar=1e-8, num_sentences=8000)
    
    if args['evaluation']:
        get_em_sparsity_from_masked_lm('bert-base-uncased', samples, att_threshold=att_threshold, hs_threshold=hs_threshold)

    if args['distribution']:
        attns, hists = get_atten_hist_from_model('bert-base-uncased', samples, att_threshold=att_threshold, hs_threshold=hs_threshold)
        attn_mask = [i.shape[-1] for i in attns]
        print(hists.shape, len(attn_mask))

        # h_state sanity check
        for i in range(10):
            print("h_state mean:{:.4f}, std:{:.4f}".format(
                np.mean(hists[0][0][i*5], axis=-1), np.std(hists[0][0][i*5], axis=-1)))

        tv.plot_atten_dist_per_token(attns, 100, sparse_hist=get_sparse_hist_token(attns, 0.0))
        tv.plot_hs_dist_per_token(hists, 100, attn_mask, scale='linear')
