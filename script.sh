#!/bin/bash
source ~/.bashrc
source activate fibottention_env

id=$1
out_dir=$2
model=$3
dataset=$4
classes=$5
device=$6
batch=$7

mkdir -p $out_dir

echo "Experiment ID: $id" >> $out_dir/log.txt
echo "Dataset: $dataset" >> $out_dir/log.txt
echo "Model: vit_${model}_patch16" >> $out_dir/log.txt
echo "Classes: $classes" >> $out_dir/log.txt
echo "Device: $device" >> $out_dir/log.txt
echo "Batch Size: $batch" >> $out_dir/log.txt
echo "Mask Ratio: $mask_ratio" >> $out_dir/log.txt
echo "----------------------------------" >> $out_dir/log.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 --master_port=$((10000 + $id)) image_classification/main_finetune.py \
    --dataset $dataset --model vit_${model}_patch16 \
    --epochs 100 \
    --cls_token \
    --nb_classes $classes \
    --batch_size $batch \
    --output_dir $out_dir \
    --log_dir $out_dir \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25
