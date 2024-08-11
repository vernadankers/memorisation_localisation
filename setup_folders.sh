#!/bin/bash

mkdir logs
mkdir logs/training
mkdir logs/gradients
mkdir logs/centroid_analysis
mkdir logs/layer_swapping
mkdir logs/layer_retraining
mkdir logs/probing

mkdir checkpoints

mkdir results
mkdir results/layer_retraining
mkdir results/layer_swapping
mkdir results/centroid_analysis
mkdir results/gradients
mkdir results/probing

for dataset in wic rte cola mrpc boolq reuters trec sst2 sst5 emotion implicithate stormfront
do
    mkdir checkpoints/${dataset}
    mkdir results/layer_retraining/${dataset}
    mkdir results/layer_swapping/${dataset}
    mkdir results/centroid_analysis/${dataset}
    mkdir results/gradients/${dataset}
    mkdir results/probing/${dataset}
done
