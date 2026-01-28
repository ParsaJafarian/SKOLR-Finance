#!/bin/bash

python run_longExp.py \
  --is_training=0 \
  --model_id=1 \
  --model=$1 \
  --data=CRSP \
  --CI \
  --train_epochs=20 \
  --dynamic_dim=32 \
  --hidden_dim=32 \
  --hidden_layers=1 \
  --seg_len=24 \
  --num_blocks=2 \
  --alpha=0.5 \
  --seq_len=24 \
  --pred_len=12
