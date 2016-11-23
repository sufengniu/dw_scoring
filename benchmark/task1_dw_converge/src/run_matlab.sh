#!/bin/bash

# configuration
emb_file=blogcatalog_10.emb
mat_file=blogcatalog.mat
input_file=../data/$emb_file
m_file=../data/$mat_file

if [ ! -d ../results ]; then
    mkdir ../results
fi

python scoring.py -m $m_file -f $input_file -s -k 10 




