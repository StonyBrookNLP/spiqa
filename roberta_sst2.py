import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from pprint import pprint
from datasets import load_dataset
import numpy as np
import pandas as pd

model_name = "textattack/roberta-base-SST-2"
#config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
num_samples = 50
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

if torch.cuda.is_available(): model = model.to("cuda")

sst2 = load_dataset("glue", "sst2")
sentences = []
labels = []
#Taking 50 instances from the SST2 dataset
for i, d in enumerate(sst2['validation']):
    if i < num_samples:
        sentences.append(d['sentence'])
        labels.append(d['label'])
    else:
        break

    
input_tokens = tokenizer.batch_encode_plus(sentences, padding=True, truncation=True, return_tensors="pt")
#pprint (input_tokens)

if torch.cuda.is_available(): 
    for i in input_tokens.keys():
        input_tokens[i] = input_tokens[i].to("cuda")
        #labels[i] = labels[i].to("cuda")

label = torch.tensor(labels).unsqueeze(0)
label = label.to("cuda")
model_output = model(**input_tokens, labels=label, output_hidden_states=True, output_attentions=True)

#Summary stat of the model_output
print ("Total items in the output tuple: ",len(model_output)) 
print ("Loss: ", model_output[0])


# print ("Number of layer representations of tokens and attention weights: ",len(model_output[1]), len(model_output[2]))
# print ("Shape of each layer representation and attention weight: ", model_output[1][0].shape, model_output[2][0].shape)

# EM score calculation
count = 0
for i, l in enumerate(labels):
    if l == torch.argmax(model_output[1][i]):
        count += 1
print("EM score: ", float(count/num_samples))

#Retrieving attentions for all layers for all instances
layer_num = 12
all_attens = []
for i in range(input_tokens["attention_mask"].shape[0]):
    num_tokens = torch.sum(input_tokens["attention_mask"][i]).item()
    total = []
    for j in range(0, layer_num):
        sentence_rep = model_output[3][j][i, :, :num_tokens, :num_tokens]
        total.append(sentence_rep)
    a = torch.stack(total, 0)
    #print(a.shape)
    all_attens.append(a)

# all_attens is a list of items where each item has the shape [#num_layers, #num_heads, #seq_len, #seq_len]
