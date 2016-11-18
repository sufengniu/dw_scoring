#!/bin/bash

data=blogcatalog
size=64
len=5
nw=10
wl=40

#source graph/convert.sh graph/$data.csv graph/$data.edgelist


for p_var in 0.25 0.5 1 2 4
do
	for q_var in 0.25 0.5 1 2 4
	do
        echo "==========================================================================================="
        echo "configuration: dimension: $size, window length: $len, number walks: $nw, walk length: $wl, p: $p_var, q: $q_var"
		python src/main.py --input graph/$data.edgelist --output emb/$data.emb --p $p_var --q $q_var --dimension $size --window-size $len --num-walks $nw --walk-length $wl
	    echo "without top-k results"
        python scoring.py # without top-k
        echo "with top-k results"
        python scoring.py -t # with top-k
        echo "==========================================================================================="
        echo "\n"
    done
done


