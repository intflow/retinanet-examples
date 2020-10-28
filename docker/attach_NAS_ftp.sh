#!/bin/bash

#sudo apt-get -y install sudo apt install nfs-common
mkdir /DL_data_big
sudo chmod -R 777 /DL_data_big

sudo apt-get install curlftpfs
curlftpfs -o allow_other intflow:intflow3121@intflow.serveftp.com:14148/DL_data_big /DL_data_big
#curlftpfs -o allow_other test_user:1q2w3e4r5t@intflow.serveftp.com:14147/NAS1 /NAS1

