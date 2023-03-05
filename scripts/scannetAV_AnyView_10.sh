#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
-- ngpus 2\
--dataset_name scannetAV \
--max_epoch 1080 \
--enc_type masked \
--enc_dropout 0.3 \
--model_name 3detr_sepview \ 
--nqueries 256 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/scannetAV_AnyView_50 \
--batchsize_per_gpu 4 \
--num_views 10
