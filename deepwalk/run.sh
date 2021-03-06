#!/bin/bash

# configuration
data=kaggle_12800
format=mat
input_file=$data.mat
input_path=../graph/*
output_file=$data.emb
length=7   # randome walk length
size=10    # embedding size
number_walk=10 # number of walks
window_size=5    # skip gram window size 
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

# run for benchmark
for nw in 10 20 50 100
do
    echo "-------- number walks = $nw ----------"
    deepwalk --format $format --input ../graph/$data.mat --output ../emb/kaggle/${data}_${nw}.emb --representation-size $size --number-walks $nw --walk-length $length --matfile-variable-name $var_name 

done

#for f in $input_path
#do

#done
