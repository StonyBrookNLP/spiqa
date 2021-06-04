'''
param_extractor: extracting parameters from RoBERTa model
'''

from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import *
from FloatingFixedToHex import ffth

import torch
import numpy as np
import random
import csv

MAX_SEQ_LEN = 320

def export_param(variables: list, file_name: str):
    try:
        for head, var in enumerate(variables):
            str_var = [['{0:0>8}'.format(ffth.float_to_hex(i)) for i in row] for row in var]
            with open(file_name + f"{head}.txt", "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(str_var)
    except TypeError:
        print('not iterable')

def extract_qkv_weights_biases(model, layer_id, ops_id='q'):
    '''
    return: weights and biases of the Q, K, V in pytorch tensor.
    weights shape: (768, 768)
    biases shape: (768,)
    '''
    ops_id_lut = {'q':'query', 'k':'key', 'v':'value'}
    weights, biases = None, None
    with torch.no_grad():
        for name, param in model.named_parameters():
            split_name = name.split('.')
            if split_name[1] == 'encoder' and \
                split_name[4] == 'attention' and \
                int(split_name[3]) == layer_id and \
                split_name[-2] == ops_id_lut[ops_id]:
                weights = param if split_name[-1] == 'weight' else weights
                biases = param if split_name[-1] == 'bias' else biases
                print(f'{split_name[-1]} of {split_name[2]} {split_name[3]} {split_name[-2]} extracted.')

    return weights, biases

def extract_attention_dense_weights_biases(model, layer_id):
    '''
    return: weights and biases of the attention output dense layer in pytorch tensor.
    weights shape: (768, 768)
    biases shape: (768,)
    '''
    weights, biases = None, None
    with torch.no_grad():
        for name, param in model.named_parameters():
            split_name = name.split('.')
            # roberta.encoder.layer.0.attention.output.dense.weight
            if split_name[1] == 'encoder' and \
                split_name[4] == 'attention' and \
                int(split_name[3]) == layer_id and \
                split_name[-3] == "output" and split_name[-2] == "dense":
                weights = param if split_name[-1] == 'weight' else weights
                biases = param if split_name[-1] == 'bias' else biases
                print(f'{split_name[-1]} of {split_name[2]} {split_name[3]} {split_name[-3]}.{split_name[-2]} extracted.')

    return weights, biases

def extract_attention_layer_inputs(inf_pipeline):
    '''
    return: inputs for all the layers (12 layer inputs, 1 last output) in the shape of
            (13, instances size, max sequence, embedding size)
    '''
    simple_text_test = [{'context': 'New York is in the United States', 'question': 'where is NYC?'}, 
                        {'context': 'ABCDEFGJH', 'question': 'what is the character after D?'}]
    predictions = qa_pipeline(simple_text_test[0], max_seq_len=MAX_SEQ_LEN)
    embd_output = predictions['hidden_states'][0][0]

    res = predictions['pipeline_prbs']
    return embd_output

def extract_qkv(inf_pipeline):
    '''
    return: q, k, and v for all the heads (12x12=144 heads) in the shape of
            [length, (12, 12, actual sequence length, embedding size)]
    '''
    simple_text_test = [{'context': 'New York is in the United States', 'question': 'where is NYC?'}, 
                        {'context': 'ABCDEFGJH', 'question': 'what is the character after D?'}]
    predictions = qa_pipeline(simple_text_test[0], max_seq_len=MAX_SEQ_LEN)

    res = predictions['pipeline_prbs']
    q_pr, k_pr, v_pr, scrs_pr, _ = predictions['pipeline_prbs']
    return np.stack(q_pr, axis=0), np.stack(k_pr, axis=0), np.stack(v_pr, axis=0)

if __name__ == '__main__':
    # model_name = "roberta-base"
    model_name = 'csarron/roberta-base-squad-v1'

    qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1)
    model = qa_pipeline.model

    weights, biases = extract_qkv_weights_biases(model, 2, 'v')
    # transform weights and biases to numpy array for the convenience
    weights = weights.detach().cpu().numpy()
    biases = biases.detach().cpu().numpy()
    weights_list = np.split(weights, 12, axis=-1)
    print(weights_list[0].shape, biases.shape)
    # save numbers:
    export_param(weights_list, "layer_2_v_weights")

    dense_weights, dense_biases = extract_attention_dense_weights_biases(model, 2)
    # transform weights and biases to numpy array for the convenience
    dense_weights = dense_weights.detach().cpu().numpy()
    dense_biases = dense_biases.detach().cpu().numpy()
    export_param([dense_weights], "layer_2_dense_weights")

    # extract input embeddings:
    embd_outputs = extract_attention_layer_inputs(qa_pipeline)

    # extract q, k, v:
    q, k, v = extract_qkv(qa_pipeline)
    print(q.shape)

    
