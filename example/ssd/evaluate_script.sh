#!/usr/bin/env bash

# GTSDB
# network="--end-epoch 210"
# network="--network mobilenet_v1 --epoch 102"
network="--network mobilenet_v2 --epoch 102"
python evaluate.py --batch-size 4 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --epoch 210 $network # --no-voc07