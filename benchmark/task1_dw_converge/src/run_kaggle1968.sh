#!/bin/bash

# configuration
format=mat
input_path=../data/kaggle
emb_path=.
length=7   # randome walk length
size=20    # embedding size
window_size=5    # skip gram window size 
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d $emb_path ]; then
    mkdir $emb_path 
fi

# run for benchmark
for nw in 10 20 50 100 200
do
    echo "-------- number walks = $nw ----------"
    deepwalk --format $format --input $input_path/1968.mat --output $emb_path/1968_${nw}.emb --representation-size $size --number-walks $nw --walk-length $length --matfile-variable-name $var_name 
    python scoring.py -m $input_path/1968.mat -f $emb_path/1968_${nw}.emb -s -k 100 -o $emb_path/1968_${nw}.mat 
done



