#!/usr/bin/env bash

# Pascal
# python train.py --train-path ./data/VOCdevkit/mxnet/train.rec --val-path ./data/VOCdevkit/mxnet/val.rec

# GTSDB
# VGG-reduced
# python train.py --batch-size 2 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0001 --lr-steps '40, 120' --end-epoch 210

# MobileNet-v1
python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.00001  --network mobilenet_v1 --pretrained ./model/mobilenet_v1 --end-epoch 102

# MobileNet-v2
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.00001 --network mobilenet_v2 --pretrained ./model/mobilenet_v2 --end-epoch 102

# MobileNet-v2-SSDLite
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68, 170' --solver 'rmsprop' --wd 0.00001 --network mobilenet_v2 --pretrained ./model/mobilenet_v2 --end-epoch 204 --lite