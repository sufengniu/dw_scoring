#!/bin/bash

# configuration
format=mat
input_path=../data/kaggle
emb_path=../result/embedding/kaggle/dw_cf
res_path=../result/mat/kaggle/dw_cf
length=7   # randome walk length
var_name=network    # mat file variable name
parallel=8    # number of parallel processor

if [ ! -d $res_path ]; then
    mkdir $res_path 
fi

for file in $input_path/*.mat
do
    filename_e=$(basename "$file") 
    filename="${filename_e%.*}"
    # run finite poly for benchmark
    for p in 1 2 3 4 5 6 7
    do
        echo "-------- finite poly $p ----------" 
        python scoring.py -m $file -f $emb_path/${filename}_finite_poly_${p}.emb -s -k 100 -o $res_path/${filename}_finite_poly_${p}.mat 
    done

    # run infinite poly for benchmark
    for p in 1 2 3 4 5 6 7
    do
        echo "-------- infinite poly $p -----------"
        python scoring.py -m $file -f $emb_path/${filename}_infinite_poly_${p}.emb -s -k 100 -o $res_path/${filename}_infinite_poly_${p}.mat
    done

done


