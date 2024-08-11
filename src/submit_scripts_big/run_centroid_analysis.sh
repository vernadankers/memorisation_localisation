#!/bin/bash

dataset=$1
modelname=$2
path_to_repo="..."

case $dataset in
  wic | cola | sst2 | mrpc | boolq | stormfront | rte)
    numlabels=2
    ;;
  sst5)
    numlabels=5
    ;;
  emotion | trec)
    numlabels=6
    ;;
  implicithate)
    numlabels=7
    ;;
  reuters)
    numlabels=8
    ;;
  *)
    echo "Unknown dataset, exiting now..."
    exit
    ;;
esac

python ../centroid_analysis.py --dataset $dataset --model_name $modelname --num_labels $numlabels \
    --checkpoint_folder ${path_to_repo}/memorisation_localisation/checkpoints
wait
