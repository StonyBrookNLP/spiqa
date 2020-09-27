# Attention Sparsity Analyzer

This repo containes python script extracting the attentions from existing
BERT-like NLP models and visualizing the sparsity of the multi-head attention.

## Case study: RoBERTa (fine-tuned for SQuAD)

### Model

| Feature                         | size           |
| ------------------------------- | -------------- |
| #layers of multi-head attention | 12             |
| #heads per layer                | 12             |
| max length of tokens            | 320            |
| attention dim                   | 320x320        |
| context+question size           | 600~700        |
| Dataset                         | squad-dev-v1.1 |

### Observation

1. Attentions have similar distribution pattern across different layers
2. Some heads have little large values (~1.0), majority of small values and large gap betweent them
3. Structured blocks of small values in the heatmap on the right side of attention matrix
   