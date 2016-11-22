#!/bin/bash

# configuration
data=blogcatalog
format=mat
input_file=../data/$data.mat
output_file=$data.emb
length=7   # randome walk length
size=10    # embedding size
number_walk=10 # number of walks
window_size=5    # skip gram window size 
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d ../results ]; then
    mkdir ../results
fi

# run for benchmark
for nw in 10 20 50 100
do
    echo "-------- number walks = $nw ----------"
    deepwalk --format $format --input $input_file --output ../data/${data}_${nw}.emb --representation-size $size --number-walks $nw --walk-length $length --matfile-variable-name $var_name 
    python scoring.py -m $input_file -f ../data/$data.emb -s -k 10 
done



