#!/usr/bin/env bash

# GTSDB
# # VGG-reduced
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network vgg16_reduced --epoch 102

# # MobileNet-v1
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v1 --epoch 102

# # MobileNet-v2
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v2 --epoch 102

# # MobileNet-v2-SSDLite
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssdlite_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v2 --epoch 204 --lite

# SqueezeNet
python train.py --batch-size 4 --train-path ./data/GTSDBdevkit/mxnet/train.rec --train-list ./data/GTSDBdevkit/mxnet/train.lst --val-path ./data/GTSDBdevkit/mxnet/val.rec --val-list ./data/GTSDBdevkit/mxnet/val.lst --prefix ./model/gtsdb_ssd  --num-class 4 --num-example 588 --data-shape-width 510 --label-width 38 --nms-topk 100 --class-names 'danger, mandatory, other, prohibitory' --no-voc07 --mean-r 125 --mean-g 127 --mean-b 130 --lr 0.0005 --lr-steps '34, 68' --solver 'rmsprop' --wd 0.0005  --network squeezenet --pretrained ./model/squeezenet_v10 --end-epoch 102

# # ResNet-18
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network resnet18 --epoch 136

# # ResNet-50
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network resnet50 --epoch 102