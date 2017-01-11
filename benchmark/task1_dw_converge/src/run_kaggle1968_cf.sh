#!/bin/bash

# configuration
format=mat
input_path=../data/kaggle
emb_path=../result/embedding/kaggle
res_path=.
length=7   # randome walk length
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d $res_path ]; then
    mkdir $res_path 
fi

# run finite poly for benchmark
for p in 1 2 3 4 5 6 7
do
    echo "-------- finite poly $p ----------" 
    python scoring.py -m ../data/kaggle/1968.mat -f $emb_path/1968_finite_poly_${p}.emb -s -k 100 -o $res_path/1968_finite_poly_${p}.mat 
done

# run infinite poly for benchmark
for p in 1 2 3 4 5 6 7
do
    echo "-------- infinite poly $p -----------"
    python scoring.py -m ../data/kaggle/1968.mat -f $emb_path/1968_infinite_poly_${p}.emb -s -k 100 -o $res_path/1968_infinite_poly_${p}.mat
done




