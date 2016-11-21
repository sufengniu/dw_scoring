#!/bin/bash

data=kaggle_3059
matfile=../graph/$data.mat
outfile=../emb/${data}_cf
size=12
iter=30
alpha=0.85
L=7


python cf.py -s $size -a $alpha -l $L -t $iter -i -m $matfile -o ../emb/kaggle/${data}_cfi.emb
python cf.py -s $size -a $alpha -l $L -t $iter -m $matfile -o ../emb/kaggle/${data}_cf.emb





#echo "==========================================================================================="
#echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, using matrix inversion"
#python cf.py -s $size -a $alpha -l $L -t $iter -i -m $data.mat
#echo "==========================================================================================="
#echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, without using matrix inversion"
#python cf.py -s $size -a $alpha -l $L -t $iter -f -m $data.mat


