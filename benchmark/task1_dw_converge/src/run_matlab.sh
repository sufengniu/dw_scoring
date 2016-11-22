#!/bin/bash

# configuration
data=blogcatalog
input_file=../data/$data.emb
matf_file=../data/$data.mat

if [ ! -d ../results ]; then
    mkdir ../results
fi

python scoring.py -m $input_file -f ../data/$data.emb -s -k 10 




