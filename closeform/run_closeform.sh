#!/bin/bash

data=blogcatalog
size=64
iter=30
alpha=0.85
L=7

echo "==========================================================================================="
echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, using matrix inversion"
python cf.py --size $size --alpha $alpha --L $L --iter $iter --inv
echo "==========================================================================================="
echo "configuration: representation dimension: $size, alpha: $alpha, length: $L, SVD iterations: $iter, without using matrix inversion"
python cf.py --size $size --alpha $alpha --L $L --iter $iter


