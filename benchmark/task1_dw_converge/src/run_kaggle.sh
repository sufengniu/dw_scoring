#!/bin/bash

# configuration
format=mat
input_path=../data/kaggle
emb_path=../result/kaggle
length=7   # randome walk length
size=10    # embedding size
number_walk=10 # number of walks
window_size=5    # skip gram window size 
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d $emb_path ]; then
    mkdir $emb_path 
fi

for file in $input_path/*.mat
do
    filename_e=$(basename "$file") 
    filename="${filename_e%.*}"
    # run for benchmark
    for nw in 10 20 50 100 200
    do
        echo "-------- number walks = $nw ----------"
        deepwalk --format $format --input $file --output $emb_path/${filename}_${nw}.emb --representation-size $size --number-walks $nw --walk-length $length --matfile-variable-name $var_name 
        python scoring.py -m $file -f $emb_path/${filename}_${nw}.emb -s -k 100 -o $emb_path/${filename}_${nw}.mat 
    done
done


