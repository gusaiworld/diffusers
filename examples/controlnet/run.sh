#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

export MODEL_DIR="./model/stable-diffusion-v1-5"
export OUTPUT_DIR="./model/demoire_4_26_st"
export HF_HUB_OFFLINE=1


accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet_1.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name='/data/guyf/1/diff/datasets/fill50k/fill50k_1.py' \
 --cache_dir='/data/guyf/1/diff/datasets/fill50k' \
 --resolution=512 \
 --learning_rate=1e-5 \
 --max_train_steps=10000 \
 --resume_from_checkpoint "latest"\
 --gradient_accumulation_steps=4 \
 --validation_image "/data/guyf/1/diff/examples/controlnet/data/image_test_part003_00000001_source.png" \
 --validation_prompt "demoire"  \
 --train_batch_size=8 \
 --mixed_precision="fp16" \
 --tracker_project_name="controlnet-demo" \
 --report_to=wandb