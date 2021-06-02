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

MAX_SEQ_LEN = 320

def extract_qkv_weights_biases(model, layer_id, ops_id='q'):
    '''
    return: weights and biases in pytorch tensor
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
    print(weights.shape, biases.shape)

    # extract input embeddings:
    embd_outputs = extract_attention_layer_inputs(qa_pipeline)

    # extract q, k, v:
    q, k, v = extract_qkv(qa_pipeline)
    print(q.shape)