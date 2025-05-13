#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

export MODEL_DIR="./model/aistable-diffusion-xl-base-1.0/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./model/demoire_5_13"
export HF_HUB_OFFLINE=1


accelerate launch --mixed_precision="fp16" --num_processes=1 --gpu_ids='0' train_controlnet_sdxl_1.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name='/data/guyf/1/diff/datasets/fill50k/fill50k_511.py' \
 --cache_dir='/data/guyf/1/diff/datasets/fill50k' \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --gradient_accumulation_steps=2 \
 --resume_from_checkpoint "latest"\
 --validation_image "/data/guyf/1/diff/examples/controlnet/data/image_test_part003_00000001_source.png" \
 --validation_prompt "demoire"  \
 --validation_steps=1000 \
 --train_batch_size=2 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --report_to=wandb

# accelerate launch train_controlnet_sdxl.py \
# --pretrained_model_name_or_path=$MODEL_DIR \
# --output_dir=$OUTPUT_DIR \
# --dataset_name=fusing/fill50k \
# --mixed_precision="fp16" \
# --resolution=1024 \
# --learning_rate=1e-5 \
# --max_train_steps=15000 \
# --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
# --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
# --validation_steps=100 \
# --train_batch_size=1 \
# --gradient_accumulation_steps=4 \
# --report_to="wandb" \
# --seed=42 \
# --push_to_hub