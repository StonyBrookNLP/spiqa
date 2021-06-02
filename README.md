# Attention Sparsity Analyzer

## Overview
This repo containes python script extracting the attentions from existing
BERT-based NLP models, visualizing the sparsity of the multi-head attention 
and pruning/quantizing the attention to see how it affects the performance.
The results are analyzed in our paper "*On the Distribution, Sparsity, and 
Inference-time Quantization of Attention Values in Transformers*" at ACL Findings 2021.

## Environment and Preparation
```
 git clone --recursive https://github.com/StonyBrookNLP/spiqa.git
 cd spiqa
 conda env create -f environment.yml
 conda activate nlp
 cd transformers
 pip install -e .
 cd ..
```

## Case study: RoBERTa (fine-tuned for SQuAD)

### Model
The model will be automatically downloaded [here](https://huggingface.co/csarron/roberta-base-squad-v1)

| Feature                         | size           |
| ------------------------------- | -------------- |
| #layers of multi-head attention | 12             |
| #heads per layer                | 12             |
| max length of tokens            | 320            |
| attention dim                   | 320x320        |
| context+question size           | 600~700        |
| Dataset                         | squad-dev-v1.1 |

### Usage
The file `roberta_squad_analyzer.py` can be used to extract and analyze the attention on the SQuAD v1.1 dataset. 
The usage is shown below:

```
usage: roberta_squad_analyzer.py [-h] [-at ATT_THRESHOLD] [-ht HS_THRESHOLD] [-d] [-e] [-m] [-s] [-qv] [-od] [-hs] [-sa SAMPLES] [-aq ATT_QUANT_BITS] [-hq HSTATE_QUANT_BITS]

roberta squad analyzer: analyzer sparsity of the roberta on squad

optional arguments:
  -h, --help            show this help message and exit
  -at ATT_THRESHOLD, --att_threshold ATT_THRESHOLD
                        set attention sparsity threshold
  -ht HS_THRESHOLD, --hs_threshold HS_THRESHOLD
                        set hidden states sparsity threshold
  -d, --distribution    print histogram
  -e, --evaluation      evaluate model only without any plot
  -m, --heatmap         print heatmap
  -s, --sparsity        compute sparsity
  -qv, --quant_visualize
                        quantize the attention
  -od, --otf_distribution
                        print attention histogram without saving aggregrated params
  -hs, --hidden_states  print hidden states histogram without saving aggregrated params
  -sa SAMPLES, --samples SAMPLES
                        number of samples for distribution
  -aq ATT_QUANT_BITS, --att_quant_bits ATT_QUANT_BITS
                        base for attention quantization
  -hq HSTATE_QUANT_BITS, --hstate_quant_bits HSTATE_QUANT_BITS
                        base for hidden states quantization
```

For example, to extract 100 instances' attention value:
```
python roberta_squad_analyzer.py -e -sa 100 
# results will be in ./params/attention_sampled.npy
```
to plot all the attention values' distribution in the dataset:
```
python roberta_squad_analyzer.py -od
# results will be in ./res_fig/at_hist_per_token_layer_N_head_M.png
```
to collect the pruned attention for 100 instances with the threshold as 0.001: 
```
python roberta_squad_analyzer.py -e -sa 100 -at 0.001
```
to collect the pruned and quantized attention for 100 instances with the threshold as 0.001 and quantization bits as 3:
```
python roberta_squad_analyzer.py -e -sa 100 -at 0.001 -aq 3
```
Their are different ways to quantize the attention:

| function call                                               | quantization method name                        | assigned value                          |
| ----------------------------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| quantize_attention_linear_slinear(attention, bits)          | uniform quantization, linear scale, w/o pruning | the closest bin edge to the original    |
| quantize_attention_linear_slinear_midval(att, bits)         | uniform quantization, linear scale, w/o pruning | mid-point of the bin edges              |
| quantize_attention_linear_slinear_clamped(att, bits)        | uniform quantization, linear scale, w/ pruning  | the closest bin edge to the original    |
| quantize_attention_linear_slinear_clamped_midval(att, bits) | uniform quantization, linear scale, w/ pruning  | mid-point of the bin edges              |
| quantize_attention_linear_slog(att, bits)                   | uniform quantization, log scale, w/o pruning    | the closest bin edge to the original    |
| quantize_attention_linear_slog_midval(att, bits)            | uniform quantization, log scale, w/o pruning    | mid-point of the bin edges              |
| quantize_attention_linear_slog_clamped_midval(att, bits)    | uniform quantization, log scale, w/ pruning     | mid-point of the bin edges              |
| quantize_attention_uniform_slinear_clamped_mean(att, bits)  | uniform quantization, linear scale, w/ pruning  | average of the original in each bin     |
| quantize_attention_uniform_slog_clamped_mean(att, bits)     | uniform quantization, log scale, w/ pruning     | average of the original in each bin     |
| quantize_attention_uniform_slog_mean(att, bits)             | uniform quantization, log scale, w/o pruning    | average of the original in each bin     |
| quantize_attention_binarization(att, bits=1)                | binarization                                    | 0 or 1.0/(number of values > threshold) |

replace the [quantization function call](https://github.com/chickenjohn/spiqa-forked-transformers/blob/8da62ec5524aa6b6e363a92c0c9ca659a812cee3/src/transformers/modeling_bert.py#L562) with the desired quantization method:
```python
# replace quantize_attention_linear_slog_clamped_midval as needed
attention_probs = self.quantize_attention_linear_slog_clamped_midval(attention_probs, quantize)
```

### Observation
1. We observed the high levels of inherent sparsity in the attention distributions, which widely exists in the heads and layers
2. most attention values can be pruned (i.e. set to zero) and the remaining non-zero values can be mapped to a small number of discrete-levels (i.e. unique values)  without any significant impact on accuracy. Approximately 80\% of the values can be set to zero without significant impact on the accuracy for QA and sentiment analysis tasks.
3. when we add quantization utilizing a log-scaling, we find a 3-bit discrete representation is sufficient to achieve accuracy within 1\% of using the full floating points of the original model.