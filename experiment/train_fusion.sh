#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$PWD" != "$DIR" ]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi


model_config=$1
tag=$2

if [ "$tag" != "" ]; then
    out_dir="${model_config%.*}-$tag-logs"
else
    out_dir="${model_config%.*}-logs"
fi


python ../bevnet/train_fusion.py \
    --model_config="$model_config" \
    --dataset_config="../dataset_configs/your_yaml" \
    --dataset_path="/path/to/dataset" \
    --output="$out_dir" \
    --batch_size=1 \
    --include_unknown \
    --log_interval=50 \
    --n_frame=3\
    --seq_len=3 \
    --frame_strides=20  \
    --resume="/path/to/resume/model" \
    --resume_epoch=0 \
    --epochs=15 \
    "${@:3}"
