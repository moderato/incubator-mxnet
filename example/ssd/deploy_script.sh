#!/usr/bin/env bash

# GTSDB
python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --epoch 210 # --network mobilenet_v1