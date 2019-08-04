#!/usr/bin/env bash

# GTSDB
# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --epoch 102

# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --network mobilenet_v1 --epoch 102

# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --network mobilenet_v2 --epoch 102

# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssdlite_ --data-shape-width 510 --network mobilenet_v2 --epoch 204 --lite

# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --network resnet18 --epoch 136

# python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --network resnet50 --epoch 102

python deploy.py --num-class 4 --prefix ./model/gtsdb_ssd_ --data-shape-width 510 --network squeezenet_v11 --epoch 4080
