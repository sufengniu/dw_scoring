#!/bin/bash

# configuration
data=cora
format=mat
input_file=../data/$data.mat
output_file=$data.emb
length=7   # randome walk length
size=16    # embedding size
number_walk=10 # number of walks
window_size=5    # skip gram window size 
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d ../result ]; then
    mkdir ../result
fi

# run for benchmark
for nw in 10 20 50 100 200
do
    #echo "-------- number walks = $nw ----------"
    #deepwalk --format $format --input $input_file --output ../result/${data}_${nw}.emb --representation-size $size --number-walks $nw --walk-length $length --matfile-variable-name $var_name --workers $parallel 
    python scoring_mat.py -m $input_file -f ../result/${data}_${nw}.emb -s -k 100 -o ../result/${data}_${nw}.mat
done



