#!/bin/bash

odtk infer /DL_data_big/ResNet18FPN_cow.pth  \
--images /DL_data_big/moodoong_cow_rbbox/task1_30000/data_500/ \
--output /DL_data_big/moodoong_cow_rbbox/rbbox/detections.json \
--rotated-bbox --resize 640 --max-size 640

