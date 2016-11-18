#!/bin/bash

data=blogcatalog
size=64
iter=30
alpha=0.85
L=7

touch $data.emb

echo "==========================================================================================="
echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, using matrix inversion"
python cf.py -s $size -a $alpha -l $L -t $iter -i
echo "==========================================================================================="
echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, without using matrix inversion"
python cf.py -s$size -a $alpha -l $L -t $iter


