#!/usr/bin/env bash

# GTSDB
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

# python demo.py --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory' --mean-r 125 --mean-g 127 --mean-b 130 --images data/GTSDBdevkit/GTSDB/JPEGImages/test/00084.jpg --thresh 0.15 --gpu 0 --network vgg16_reduced --epoch 210

# python demo.py --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory' --mean-r 125 --mean-g 127 --mean-b 130 --images data/GTSDBdevkit/GTSDB/JPEGImages/test/00084.jpg --thresh 0.15 --gpu 0 --network mobilenet_v1 --epoch 102

# python demo.py --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory' --mean-r 125 --mean-g 127 --mean-b 130 --images data/GTSDBdevkit/GTSDB/JPEGImages/test/00084.jpg --thresh 0.15 --gpu 0 --network mobilenet_v2 --epoch 102

python demo.py --prefix ./model/gtsdb_ssdlite_ --data-shape-width 510 --class-names 'danger, mandatory, other, prohibitory' --mean-r 125 --mean-g 127 --mean-b 130 --images data/GTSDBdevkit/GTSDB/JPEGImages/test/00084.jpg --thresh 0.15 --gpu 0 --network mobilenet_v2 --epoch 204 --lite