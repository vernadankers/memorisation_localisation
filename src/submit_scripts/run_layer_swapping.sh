#!/bin/bash

dataset=$1
model=$2
path_to_repo="..."

seed=1
for freeze in "embeddings-0-1-2-3-4-5-6-7-8-9" "embeddings-0-1-2-3-4-7-8-9-10-11" "embeddings-2-3-4-5-6-7-8-9-10-11"
do
    python ../layer_swapping.py \
        --dataset $dataset \
        --custom_model "${path_to_repo}/checkpoints/${dataset}/${model}_seed=${seed}_freeze=${freeze}_epoch=50.pt" \
        --model_name $model \
        --model_seed $seed
    wait
done

freeze="embeddings"
for seed in 1 2 3
do
    python ../layer_swapping.py \
        --dataset $dataset \
        --custom_model "${path_to_repo}/checkpoints/${dataset}/${model}_seed=${seed}_freeze=${freeze}_epoch=50.pt" \
        --model_name $model \
        --model_seed $seed
    wait
done
