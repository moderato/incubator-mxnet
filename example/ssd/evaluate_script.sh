#!/usr/bin/env bash

# GTSDB
python evaluate.py --batch-size 1 --rec-path ./data/GTSDBdevkit/mxnet/val.rec --list-path ./data/GTSDBdevkit/mxnet/val.lst --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory'  --mean-r 125 --mean-g 127 --mean-b 130 --epoch 210 # --network mobilenet_v1 --epoch 102 # --no-voc07