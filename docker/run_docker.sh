#!/bin/bash
#X11

#sudo bash attach_NAS_ftp.sh
sudo xhost +local:root

#Mount Data folders
sudo mkdir /DL_data_big
sudo mount 192.168.0.18:/DL_data_big /DL_data_big
#sudo mount 192.168.0.14:/NAS1 /NAS1

#Pull update docker image
sudo docker pull intflow/odtk:melting_cow

#Run Dockers for CenterNet+DeepSORT
sudo docker run --name odtk_melting_cow \
--gpus all --rm -p 2444:2444 \
--mount type=bind,src=/home/intflow/works,dst=/works \
--mount type=bind,src=/DL_data_big,dst=/DL_data_big \
--net=host \
--privileged \
--ipc=host \
-it intflow/odtk:melting_cow /bin/bash


#-it intflow/gc2020_intflow_track4:default /bin/bash
#-it nvcr.io/nvidia/pytorch:20.03-py3 /bin/bash

#-it nvcr.io/nvidia/tensorrt:20.07-py3 /bin/bash

