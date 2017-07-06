#!/usr/bin/env bash
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/media/tjiang/Elements/ImageClassification/Deep_Fashion/lenet-model

# Where the dataset is saved to.
DATASET_DIR=/media/tjiang/Elements/ImageClassification/Deep_Fashion/

# Run training.
#python train_deep_fashion.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=lenet \
#  --max_number_of_steps=5000 \
#  --batch_size=50 \
#  --train_image_size=100 \
#  --learning_rate=0.005 \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \
#  --optimizer=sgd \
#  --learning_rate_decay_type=fixed \
#  --weight_decay=0

# Run evaluation.
python eval_deep_fashion.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=lenet
