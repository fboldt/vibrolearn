#!/bin/bash

# Run the experiment with the specified parameters
if [ $# -ne 1 ]; then
    N=1
else
    N=$1
fi
for i in {1..$N}; do
    for setup in dataset/cwru/*.json; do
        echo "Running experiment with setup: $setup, iteration: $i"
        python main.py -e $setup  #> results/$(basename $setup .json)_${i}.txt 
        python main.py -e $setup -a  #> results/$(basename $setup .json)_${i}_aug.txt
    done
done
