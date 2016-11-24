#!/bin/bash

# configuration
emb_file=345_infinite_poly_1.emb
mat_file=345.mat
input_file=../result/embedding/kaggle/$emb_file
m_file=../data/kaggle/$mat_file

if [ ! -d ../result ]; then
    mkdir ../result
fi

python scoring.py -m $m_file -f $input_file -s -k 100 





