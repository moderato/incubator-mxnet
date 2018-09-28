#!/usr/bin/env bash

# Pascal
# python train.py --train-path ./data/VOCdevkit/mxnet/train.rec --val-path ./data/VOCdevkit/mxnet/val.rec

# GTSDB
# network="--end-epoch 210"
# network="--network mobilenet_v1 --pretrained ./model/mobilenet_v1 --end-epoch 102"
network="--network mobilenet_v2 --pretrained ./model/mobilenet_v2 --end-epoch 102"
python train.py --batch-size 2 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --lr 0.0001 --lr-steps '40, 120' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 $network