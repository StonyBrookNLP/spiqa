#!/bin/bash
DATA_FILE="sparse_transformer_quantization_attention_data.tar.bz2"
if [ ! -f "$DATA_FILE" ]; then
    echo "Downloading data..."
    wget https://compas.cs.stonybrook.edu/downloads/sparse_transformer/sparse_transformer_quantization_attention_data.tar.bz2
fi

echo "Extracting files..."
tar -xf "$DATA_FILE"