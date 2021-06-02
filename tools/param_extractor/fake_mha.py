"""
dependency: ffth: https://compas.cs.stonybrook.edu:1121/tianchu.ji/BasicElementsProj/tree/master/proj/FloatingFixedToHex
numpy, pytorch
"""

import numpy as np
import math, csv
import torch

from FloatingFixedToHex import ffth

num_heads = 12
hidden_states = 768
hidden_states_per_head = hidden_states / num_heads
embd_size = 768
seq_len = 320

def softmax1d(arr, axis=-1):
    t = torch.Tensor(arr).type(torch.float32)
    ret = torch.nn.Softmax(dim=-1)(t)
    ret = ret.numpy()
    return ret


def export_param(varnames: list, variables: list):
    try:
        for varname, var in zip(varnames, variables):
            str_var = [['{0:0>8}'.format(ffth.float_to_hex(i)) for i in row] for row in var]
            with open("{}.csv".format(varname), "w+", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(str_var)
    except TypeError:
        print('not iterable')

# random generated params and inputs
Wq, Wk, Wv = (np.random.random((embd_size, hidden_states)).astype('float32') for i in range(3))
Bq, Bk, Bv = (np.random.random((1, hidden_states)).astype('float32') for i in range(3))

export_param(['Wq', 'Wk', 'Wv', 'Bq', 'Bk', 'Bv'], [Wq, Wk, Wv, Bq, Bk, Bv])

embd_inputs = np.random.random((seq_len, embd_size))
Q = np.matmul(embd_inputs, Wq) + Bq
K = np.matmul(embd_inputs, Wk) + Bk
V = np.matmul(embd_inputs, Wv) + Bv

Qh = np.split(Q, 2, axis=-1)
Kh = np.split(K, 2, axis=-1)
Vh = np.split(V, 2, axis=-1)

export_param(['inputs'], [embd_inputs])
export_param(('Qh_0', 'Qh_1', 'Kh_0', 'Kh_1', 'Vh_0', 'Vh_1'), \
                (Qh[0], Qh[1], Kh[0], Kh[1], Vh[0], Vh[1]))

att_score = []
for q, k in zip(Qh, Kh):
    curr_score = np.matmul(q, np.transpose(k, axes=(1, 0)))
    curr_score = curr_score / math.sqrt(hidden_states_per_head)
    att_score.append(curr_score)

export_param(('att_score_0', 'att_score_1'), att_score)

att_prob = []
for att in att_score:
    att_prob.append(softmax1d(att))

export_param(('att_weight_0', 'att_weight_1'), att_prob)

Z = []
for att_prob_head, v in zip(att_prob, Vh):
    Z.append(np.matmul(att_prob_head, v))

export_param(['Z_0', 'Z_1'], Z)

print(len(Z))