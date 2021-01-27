"""
Analyzing BERT-like model pretrained for SST2
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset
import argparse as ag
import numpy as np
import pandas as pd
import transformer_visualization as tv

import os, random

PARAM_PATH = './params/'
ATT_SIZE = [12, 12]

def extract_sst2(model_name, num_sentences):
    sst2 = load_dataset("glue", "sst2", cache_dir='./data')['validation']
    sentences = []
    labels = []
    #Taking some instances from the SST2 dataset
    if num_sentences < 0:
        print("extracting all {} samples...".format(sst2.num_rows))
        selected_data_id = list(range(sst2.num_rows))
    else:    
        selected_data_id = np.random.choice(sst2.num_rows, num_sentences, replace=False).tolist()
    
    for d in selected_data_id:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_se = tokenizer(sst2[d]['sentence'], return_tensors='np')['input_ids']
        if (tokenized_se.shape[-1] < tokenizer.max_len) :
            sentences.append(sst2[d]['sentence'])
            labels.append(sst2[d]['label'])
        else:
            break
    
    print("actual selected: ", len(sentences))
    return sentences, labels

def run_model(model_name, input_tokens, labels, att_threshold=0.0, hs_threshold=0.0, quantize_att_bits=0.0, quantize_hstat_bits=0.0, device="cuda"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model = model.to(device)

    label = torch.tensor(labels).unsqueeze(0)
    label = label.to(device)


    head_mask = np.ones(ATT_SIZE)
    head_mask[0][9], head_mask[0][11], head_mask[1][2], head_mask[7][8] = 0, 0, 0, 0
    head_mask = None

    with torch.no_grad():
        model_output = model(**input_tokens, labels=label, output_hidden_states=True, output_attentions=True, \
                            att_threshold=att_threshold, hs_threshold=hs_threshold, head_mask=head_mask, \
                            quantize_att_bits=quantize_att_bits, quantize_hstat_bits=quantize_hstat_bits)

    #Summary stat of the model_output
    print ("Total items in the output tuple: ",len(model_output)) 
    print ("Loss: ", model_output[0])

    #Retrieving attentions for all layers for all instances
    all_attens = []
    for i in range(input_tokens["attention_mask"].shape[0]):
        num_tokens = torch.sum(input_tokens["attention_mask"][i]).item()
        total = []
        for j in range(len(model_output[3])):
            sentence_rep = model_output[3][j][i, :, :num_tokens, :num_tokens]
            total.append(sentence_rep)
        a = (torch.stack(total, 0)).to('cpu').numpy()
        all_attens.append(a)

    # EM score calculation
    count = 0
    for i, l in enumerate(labels):
        if l == torch.argmax(model_output[1][i]):
            count += 1
    print("EM score: ", float(count/len(all_attens)))

    return model_output[0].item(), all_attens, count

def get_atten_per_token(model_name, num_sentences, att_threshold=0.0, hs_threshold=0.0, device="cuda", stored_attentions=False):
    param_file_path = PARAM_PATH + model_name
    head_mask = None

    attentions, loss = None, 0.0
    if os.path.isfile(param_file_path + "_attention.npy") and stored_attentions:
        print("loading parameters from file...")
        with open(param_file_path + "_attention_mask.npy", "rb") as att_mask_file:
            attn_mask = np.load(att_mask_file, allow_pickle=True)
            loss = np.load(att_mask_file, allow_pickle=True)[0]
        with open(param_file_path + "_attention.npy", "rb") as att_file:
            attentions = [np.load(att_file, allow_pickle=True) for i in range(len(attn_mask))]
            
    else:
        # get sentences
        sentences, labels = extract_sst2(model_name, num_sentences)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_tokens = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors="pt")
        attn_mask = input_tokens['attention_mask'].to('cpu').numpy()
        for i in input_tokens.keys():
            input_tokens[i] = input_tokens[i].to(device)

        # run model
        loss, attentions, _ = \
            run_model(model_name, input_tokens, labels, att_threshold=att_threshold, hs_threshold=hs_threshold)

        if stored_attentions:
            with open(param_file_path + "_attention_mask.npy", "wb+") as att_mask_file:
                np.save(att_mask_file, attn_mask)
                np.save(att_mask_file, np.array([loss]))
            with open(param_file_path + "_attention.npy", "wb+") as att_file:
                for i in range(len(attn_mask)): np.save(att_file, attentions[i], allow_pickle=False)
    
    for i, att in enumerate(attentions):
        print ("Shape of attention weight matrices", i, att.shape)
    return loss, attentions

def get_sparse_hist_token(attn, offset, sparsity_bar=0.0):
    all_sparse_count = None
    for att in attn:
        curr_sparse_count = np.apply_along_axis(lambda a: float((a <= (sparsity_bar + offset)).sum()) / att.shape[-1], -1, att)
        all_sparse_count = curr_sparse_count if all_sparse_count is None \
                                else np.concatenate((curr_sparse_count, all_sparse_count), axis=-1)
    
    sparse_hist = np.apply_along_axis(lambda a: np.histogram(a, bins=10, range=(0.0, 1.0))[0], -1, all_sparse_count)
    sparse_hist = np.apply_along_axis(lambda a: a / np.sum(a), -1, sparse_hist)
    return sparse_hist


def get_em_sparsity_from_sa(model_name: str, num_sentences: int, att_threshold=0.0, hs_threshold=0.0, att_quant_bits = 0.0, hstate_quant_bits = 0.0, device='cuda'):
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
        res, total_elem_count, inst_count = None, 0, 0

        # fetch data
        all_input_tokens, all_labels = extract_sst2(model_name, num_sentences)
        batch_size=20
        all_input_tokens = list(chunks(all_input_tokens, batch_size))
        all_labels = list(chunks(all_labels, batch_size))

        for batch_inputs, batch_labels in zip(all_input_tokens, all_labels):
            inst_count += len(batch_inputs)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            input_tokens = tokenizer.batch_encode_plus(batch_inputs, padding=True, truncation=True, return_tensors="pt")
            for i in input_tokens.keys():
                input_tokens[i] = input_tokens[i].to(device)

            loss, attentions, match_count = run_model(model_name, input_tokens, batch_labels, \
                        att_threshold=att_threshold, hs_threshold=hs_threshold, \
                        quantize_att_bits=att_quant_bits, quantize_hstat_bits=hstate_quant_bits,  device=device)
                
            def get_spars(x, axis): 
                return x.shape[-1] ** 2 - np.count_nonzero(x[:, :, :, :], axis=axis)
            def agg_func(f): return np.stack([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            def add_func(f): return np.sum([f(i, axis=(-2, -1)) for i in attentions], axis=0)
            if res is None:
                res = {'score': match_count, 'mean': agg_func(np.mean),
                    'max': agg_func(np.amax), 'min': agg_func(np.amin), 
                    'std': agg_func(np.std), 'sparsity': add_func(get_spars)}
            else:
                res['score'] += match_count
                res['max'] = np.concatenate((res['max'], agg_func(np.amax)), axis=0)
                res['min'] = np.concatenate((res['min'], agg_func(np.amin)), axis=0)
                res['mean'] = np.concatenate((res['mean'], agg_func(np.mean)), axis=0)
                res['std'] = np.concatenate((res['std'], agg_func(np.std)), axis=0)
                res['sparsity'] = np.add(res['sparsity'], add_func(get_spars))

            total_elem_count += sum([att.shape[-1] * att.shape[-1] for att in attentions])

        res['sparsity'] = res['sparsity'].astype(float) / total_elem_count
        res['score'] /= float(inst_count)

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
    arg_parser.add_argument("-aq", "--att_quant_bits", default=0.0,
                            required=False, help="base for attention quantization")
    arg_parser.add_argument("-hq", "--hstate_quant_bits", default=0.0,
                            required=False, help="base for hidden states quantization")
    
    args = vars(arg_parser.parse_args())
    att_threshold = float(args['att_threshold'])
    hs_threshold = float(args['hs_threshold'])
    att_quant_bits = float(args['att_quant_bits'])
    hstate_quant_bits = float(args['hstate_quant_bits'])
    samples = int(args['samples'])

    if args['evaluation']:
        get_em_sparsity_from_sa('textattack/roberta-base-SST-2', samples, att_threshold=att_threshold, hs_threshold=hs_threshold, att_quant_bits=att_quant_bits, hstate_quant_bits=hstate_quant_bits, device='cuda')

    if args['distribution']:
        loss, attns = get_atten_per_token('textattack/roberta-base-SST-2', samples, att_threshold=att_threshold, hs_threshold=hs_threshold, stored_attentions=True)
        tv.get_diversity(attns, 100, model_name='roberta-base-SST-2')
        tv.plot_atten_dist_per_token(attns, 100, sparse_hist=get_sparse_hist_token(attns, 1e-8), ylim=(0.2, 1), model_name='roberta-base-SST-2')
