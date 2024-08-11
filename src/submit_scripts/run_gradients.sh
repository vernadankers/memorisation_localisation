#!/bin/bash

dataset=$1
model=$2
path_to_repo="..."

for seed in 1 2 3
do
    for freeze in "embeddings"
    do
        python ../gradient_analysis.py --dataset $dataset --model_seed $seed \
         --custom_model "${path_to_repo}/checkpoints/${dataset}/${model}_seed=${seed}_freeze=${freeze}_epoch=50.pt" \
         --model_name $model
        wait
  done
done

seed=1
for freeze in "embeddings-0-1-2-3-4-5-6-7-8-9-10-11-fullfreeze"
do
    python ../gradient_analysis.py --dataset $dataset --model_seed $seed \
     --custom_model "${path_to_repo}/checkpoints/${dataset}/${model}_seed=${seed}_freeze=${freeze}_epoch=50.pt" \
     --model_name $model
    wait
done
