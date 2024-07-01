#!/bin/bash
source ~/.bashrc

id=$1
out_dir=$2
model=$3
dataset=$4
classes=$5
device=$6
batch=$7
mask_ratio=$8

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1  image_classification/main_finetune.py  \
    --dataset $dataset --model vit_${model}_patch16 \
    --dist_url 'tcp://localhost:1000'$id \
    --epochs 100 \
    --cls_token \
    --nb_classes $classes \
    --batch_size $batch \
    --output_dir $out_dir \
    --log_dir $out_dir \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mask_ratio $mask_ratio