#!/bin/bash

partition="..."

for model in "facebook_opt-125m" "EleutherAI_pythia-160m-deduped" "EleutherAI_gpt-neo-125m" "bert-base-cased"
do
    for dataset in "mrpc" "wic" "rte" "cola" "boolq" "sst2" "sst5" "emotion" "trec" "reuters" "stormfront" "implicithate"
    do
        echo $1
        if [[ "${1}" == "training" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
               -o ../../logs/training/${dataset}_${model}.out \
               run_training.sh $dataset $model
        elif [[ "${1}" == "swapping" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
                 -o ../../logs/layer_swapping/${dataset}_${model}.out \
                 run_layer_swapping.sh $dataset $model
        elif [[ "${1}" == "retraining" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
                 -o ../../logs/layer_retraining/${dataset}_${model}.out \
                 run_layer_retraining.sh $dataset $model
        elif [[ "${1}" == "centroid_analysis" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
                 -o ../../logs/centroid_analysis/${dataset}_${model}.out \
                 run_centroid_analysis.sh $dataset $model
        elif [[ "${1}" == "gradients" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
                 -o ../../logs/gradients/${dataset}_${model}.out \
                 run_gradients.sh $dataset $model
        elif [[ "${1}" == "probing" ]]
        then
            sbatch -t 48:00:00  --gres=gpu:1 -p $partition \
                  -o ../../logs/probing/${dataset}_${model}.out \
                  run_probing.sh $dataset $model
        fi
    done
done
