#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Set the CUDA_VISIBLE_DEVICES to the specified GPUs (GPUS should be a comma-separated string, e.g., "0,1,2")
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Ensure GPUS is passed as an integer to nproc_per_node
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    $(dirname "$0")/val.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch ${@:4}


# bash tools/dist_val.sh configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_brickkilns.py \
#     work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_brickkilns1/epoch_36.pth \
#     1,2


# Train - SWINIR Bihar, Test - SWINIR Test Bihar
# bash tools/dist_val.sh configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_bihar_to_bihar.py \
#     work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_bihar_to_bihar/epoch_50.pth \
#     1,2


# Train - SWINIR Haryana, Test - SWINIR Test Bihar
# bash tools/dist_val.sh configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar.py \
#     work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_swinir_haryana_to_bihar/epoch_50.pth \
#     1,2


# Train - Haryana, Test - Test Bihar
# bash tools/dist_val.sh configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_haryana.py \
#     work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_haryana/epoch_50.pth \
#     1,2