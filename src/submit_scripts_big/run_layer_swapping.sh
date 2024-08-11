#!/bin/bash

dataset=$1
model=$2
path_to_repo="..."

for seed in 1
do
  for freeze in "embeddings"
  do
        python ../layer_swapping.py \
            --dataset $dataset \
            --custom_model "${path_to_repo}/checkpoints/${dataset}/${model}_seed=${seed}_freeze=${freeze}_epoch=50.pt" \
            --model_name $model \
            --model_seed $seed \
            --checkpoint_folder ${path_to_repo}/checkpoints
        wait
  done
done
