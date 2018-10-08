#!/usr/bin/env bash

# GTSDB
# VGG-reduced
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network vgg16_reduced --epoch 210

# MobileNet-v1
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v1 --epoch 102

# MobileNet-v2
# python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v2 --epoch 102

# MobileNet-v2-SSDLite
python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssdlite_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --no-voc07 --network mobilenet_v2 --epoch 204 --lite