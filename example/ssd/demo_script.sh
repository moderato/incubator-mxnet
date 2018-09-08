#!/usr/bin/env bash

# GTSDB
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python demo.py --network vgg16_reduced --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory' --mean-r 125 --mean-g 127 --mean-b 130 --epoch 102 --images data/GTSDBdevkit/GTSDB/JPEGImages/test/00084.jpg --thresh 0.15 --gpu 0 --network mobilenet_v1