#!/usr/bin/env bash

# GTSDB
# network="--epoch 210"
# network="--network mobilenet_v1 --epoch 102"
network="--network mobilenet_v2 --epoch 102"
python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 $network