#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    main-ndp.py --model_name_or_path THUDM/chatglm-6b  --data_path Smart/Q_A \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size  1 \
   --gradient_accumulation_steps 1 --lora_dim 128  --zero_stage $ZERO_STAGE \
   --output_dir $OUTPUT &> $OUTPUT/training.log \
   --quantization_bit 8
