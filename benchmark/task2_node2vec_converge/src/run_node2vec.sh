#/bin/bash

data=1968
size=20
len=5
wl=7
emb_path=emb
res_path=../results
p_var=1 # 0.25 0.5 1 2 5
q_var=1 # 0.25 0.5 1 2 5

if [ ! -d emb ]; then
    mkdir emb
fi
#source graph/convert.sh graph/$data.csv graph/$data.edgelist

for nw in 10 20 50 100 200
do
    echo "==========================================================================================="
    echo "configuration: dimension: $size, window length: $len, number walks: $nw, walk length: $wl, p: $p_var, q: $q_var"
    python src/main.py --input ../data/$data.edgelist --output $emb_path/${data}_${nw}.emb --p $p_var --q $q_var --dimension $size --window-size $len --num-walks $nw --walk-length $wl
    echo "without top-k results"
    python scoring.py -m ../data/$data.mat -f $emb_path/${data}_${nw}.emb -s -k 100 -o ${res_path}/${data}_${nw}.mat # without top-k
    echo "==========================================================================================="
    echo ""
done

