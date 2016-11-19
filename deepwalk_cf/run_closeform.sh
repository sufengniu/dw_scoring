#!/bin/bash

data=kaggle_2365
matfile=../graph/$data.mat
outfile=../emb/${data}_cf
size=64
iter=30
alpha=0.85
L=7

for nw in 10 20 50 100
do
    python cf.py -s $size -a $alpha -l $L -t $iter -i -m $matfile -o ../emb/${data}_cf_${nw}.emb
    python cf.py -s $size -a $alpha -l $L -t $iter -m $matfile -o ../emb/${data}_cf_i_${nw}.emb
done




#echo "==========================================================================================="
#echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, using matrix inversion"
#python cf.py -s $size -a $alpha -l $L -t $iter -i -m $data.mat
#echo "==========================================================================================="
#echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, without using matrix inversion"
#python cf.py -s $size -a $alpha -l $L -t $iter -f -m $data.mat


