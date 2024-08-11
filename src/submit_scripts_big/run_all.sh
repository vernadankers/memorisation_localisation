#!/bin/bash

for model in "facebook_opt-1.3b"
do
    for dataset in wic boolq cola mrpc emotion sst2 sst5 trec implicithate stormfront reuters rte
    do
        if [[ "${1}" == "training" ]]
        then
            sbatch -t 8:00:00 -p ampere --gres=gpu:1 -N 1 -n 1 \
                   -o ~/memorisation_localisation/logs/training/${dataset}_${model}.out \
                   run_training.sh $dataset $model
        elif [[ "${1}" == "swapping" ]]
        then
            sbatch -t 8:00:00 -p $partition --gres=gpu:1 -N 1 -n 1 \
                   -o ~/memorisation_localisation/logs/layer_swapping/${dataset}_${model}.out \
                   run_layer_swapping.sh $dataset $model
        elif [[ "${1}" == "centroid_analysis" ]]
        then
            sbatch -t 8:00:00 -p $partition --gres=gpu:1 -N 1 -n 1 \
                   -o ~/memorisation_localisation/logs/centroid_analysis/${dataset}_${model}.out \
                   run_centroid_analysis.sh $dataset $model
        fi
    done
done
