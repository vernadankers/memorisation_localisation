#!/bin/bash

dataset=$1
modelname=$2

case $dataset in
  wic | cola | sst2 | mrpc | boolq | stormfront | rte | emotionv2 | trecv2 | sst5v2 | implicithatev2 | reutersv2)
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

python ../centroid_analysis.py --dataset $dataset --model_name $modelname --num_labels $numlabels --setup memorised
wait
python ../centroid_analysis.py --dataset $dataset --model_name $modelname --num_labels $numlabels --setup mixt
wait
python ../centroid_analysis2.py --dataset $dataset --model_name $modelname --num_labels $numlabels --setup mixt
wait
