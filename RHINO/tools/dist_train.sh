#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29510}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Set the CUDA_VISIBLE_DEVICES to the specified GPUs (GPUS should be a comma-separated string, e.g. "0,2,3")
export CUDA_VISIBLE_DEVICES=$GPUS

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Ensure GPUS is passed as an integer to nproc_per_node, not as a string like "0,1"
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nohup python3 -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} > my_output.log 2>&1 &


# bash tools/dist_train.sh 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_combined.py' 2,3

# bash tools/dist_train.sh 'configs-mine/rhino-resnet/rhino_phc_haus-4scale_r50_2xb2-36e_bihar.py' 1,2

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_bihar.py' 2,3

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_m0.py' 2,3

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_haryana.py' 2

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_swinir_bihar.py' 1,2

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_swinir_haryana.py' 1,2

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_thera_bihar.py' 1,2,3

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_thera_haryana.py' 2,3

# bash tools/dist_train.sh 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_sentinel.py' 0