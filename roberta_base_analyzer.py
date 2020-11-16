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

def run_mask_fill_pipeline(model_name: str, inst: str, masked_idx: int, att_threshold=0.0, hs_threshold=0.0, head_mask=None):
    mf_pipeline = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=0, \
        topk=20, att_threshold=att_threshold, hs_threshold=hs_threshold, \
        output_attentions=True, output_hidden_states=True)

    # fetch data:
    masked_insts, correct_token = [], []

    word_list = inst.split(' ')
    correct_token = word_list[masked_idx]
    word_list[masked_idx] = '[MASK]'
    masked_inst = " ".join(word_list)

    results = mf_pipeline([masked_inst])
    
    correctness = False
    for possible_res in results['result'][0]:
        if possible_res['token_str'].lower() == correct_token.lower():
            correctness = True
            break

    return correctness, results['hidden_states'], results['attentions']
    

def get_atten_hist_from_model(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0):
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

def get_em_sparsity_from_masked_lm(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0):
    
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
        insts = extract_inst_wikipedia(num_sentences)
        random.seed(13312)
        masked_ids = [random.randint(0, len(i.split(' '))-1) for i in insts]

        head_mask=None

        glb_counter, correct_counter = 0, 0
        res, total_elem_count = None, 0
        for inst, masked_id in zip(insts, masked_ids):
            print("running pipeline iter {}/{}".format(glb_counter+1, num_sentences))
            correctness, hidden_states, attentions = \
                run_mask_fill_pipeline(model_name, inst, masked_id, \
                    att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask)
            if correctness: correct_counter += 1
            glb_counter += 1
            em_score = float(correct_counter) / glb_counter

            def get_spars(x, axis): 
                return x.shape[-1] ** 2 - np.count_nonzero(x[:, :, :, :], axis=axis)
            def agg_func(f): return np.stack([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            def add_func(f): return np.sum([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            if res is None:
                res = {'score': em_score, 'mean': agg_func(np.mean),
                    'max': agg_func(np.amax), 'min': agg_func(np.amin), 
                    'std': agg_func(np.std), 'sparsity': add_func(get_spars)}
            else:
                res['score'] = em_score
                res['max'] = np.concatenate((res['max'], agg_func(np.amax)), axis=0)
                res['min'] = np.concatenate((res['min'], agg_func(np.amin)), axis=0)
                res['mean'] = np.concatenate((res['mean'], agg_func(np.mean)), axis=0)
                res['std'] = np.concatenate((res['std'], agg_func(np.std)), axis=0)
                res['sparsity'] = np.add(res['sparsity'], add_func(get_spars))

            total_elem_count += sum([att.shape[-1] * att.shape[-1] for att in attentions])

        res['sparsity'] = res['sparsity'].astype(float) / total_elem_count

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
        get_em_sparsity_from_masked_lm("bert-base-uncased", samples, att_threshold=att_threshold, hs_threshold=hs_threshold)

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
