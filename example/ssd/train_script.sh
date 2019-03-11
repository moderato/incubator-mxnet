#!/usr/bin/env bash

# Pascal
# python train.py --train-path ./data/VOCdevkit/mxnet/train.rec --val-path ./data/VOCdevkit/mxnet/val.rec

# GTSDB
# # VGG-reduced
# python train.py --batch-size 2 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0001 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.00001 --end-epoch 102

# # MobileNet-v1
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.00001  --network mobilenet_v1 --pretrained ./model/mobilenet_v1 --end-epoch 102

# # MobileNet-v2
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.00001 --network mobilenet_v2 --pretrained ./model/mobilenet_v2 --end-epoch 102

# # MobileNet-v2-SSDLite
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68, 170' --solver 'rmsprop' --wd 0.00001 --network mobilenet_v2 --pretrained ./model/mobilenet_v2 --end-epoch 204 --lite

# # SqueezeNet-v1.0
# python train.py --batch-size 16 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.001 --lr-steps '408, 612' --solver 'rmsprop' --wd 0.0005  --network squeezenet_v10 --pretrained ./model/squeezenet_v10 --end-epoch 816

# SqueezeNet-v1.1
python train.py --batch-size 16 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.001 --lr-steps '408, 612' --solver 'rmsprop' --wd 0.0005  --network squeezenet_v11 --pretrained ./model/squeezenet_v11 --end-epoch 816

# # ResNet-18
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '68, 102' --solver 'rmsprop' --wd 0.0005  --network resnet18 --pretrained ./model/resnet-18 --end-epoch 136

# # ResNet-50
# python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0001 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.0005  --network resnet50 --pretrained ./model/resnet-50 --end-epoch 102