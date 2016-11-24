#!/bin/bash

# configuration
emb_file=blogcatalog_finite_poly_2.emb
mat_file=blogcatalog.mat
input_file=../result/embedding/blogcatalog/$emb_file
m_file=../data/$mat_file

if [ ! -d ../result ]; then
    mkdir ../result
fi

python scoring.py -m $m_file -f $input_file -s -k 10 




