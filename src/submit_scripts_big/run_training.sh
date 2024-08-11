#!/bin/bash

dataset=$1
modelname=$2
bsz=32
aux="rte"

for seed in 1
do
    # python ../main.py --training_type regular --dataset $dataset --freeze embeddings --model_name $modelname \
    #     --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 5e-6 --checkpoint_folder \
    #     /home/co-dank1/rds/hpc-work/memorisation_localisation/checkpoints
    #     #/home/s2112866/memorisation_localisation/checkpoints
    #
    # wait
    python ../main.py --training_type regular --dataset $dataset --freeze embeddings --model_name $modelname \
        --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 5e-6 --normal --checkpoint_folder \
        /home/co-dank1/rds/hpc-work/memorisation_localisation/checkpoints
        #/home/s2112866/memorisation_localisation/checkpoints
    wait
    # if [ "${1}" == "trec" ]; then
    #     python ../main.py --training_type ratios --dataset $dataset --freeze embeddings --model_name $modelname \
    #         --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5
    #     wait
    # fi
done

# for seed in 1
# do
#     if [ "${1}" != "rte" ]; then
#         python ../main.py --training_type freeze_train --dataset $dataset --freeze embeddings 2 3 4 5 6 7 8 9 10 11  --model_name $modelname \
#             --aux_datasets $aux --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5 --normal
#         wait
#         python ../main.py --training_type freeze_train --dataset $dataset --freeze embeddings 0 1 2 3 4 7 8 9 10 11 --model_name $modelname \
#             --aux_datasets $aux --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5 --normal
#         wait
#         python ../main.py --training_type freeze_train --dataset $dataset --freeze embeddings 0 1 2 3 4 5 6 7 8 9 --model_name $modelname \
#             --aux_datasets $aux --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5 --normal
#         wait
#         # python ../main.py --training_type freeze_train --dataset $dataset --freeze embeddings 0 1 2 3 4 5 6 7 8 9 10 11 --model_name $modelname \
#         #     --aux_datasets $aux --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5
#         # wait
#     fi
#     # if [ "${1}" == "rte" ]; then
#     #     python ../main.py --training_type freeze_train --dataset $dataset --freeze embeddings 0 1 2 3 4 5 6 7 8 9 10 11 --model_name $modelname \
#     #         --aux_datasets "wic" --num_epochs 50 --batch_size $bsz --model_seed $seed --lr 1e-5
#     #     wait
#     # fi
#     # python ../main.py --training_type mmap --dataset $dataset --freeze embeddings --model_name $modelname \
#     #     --aux_datasets $aux --num_epochs 10 --batch_size $bsz --model_seed $seed --lr 1e-5
#     # wait
# done
