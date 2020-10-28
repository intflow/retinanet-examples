#!/bin/bash

sudo apt-get -y install nfs-common
mkdir /DL_data_big
mount 192.168.0.18:/volume1/DL_data_big /DL_data_big


#sudo mkdir /NAS1

#sudo chmod -R 777 /DL_data
#sudo chmod -R 777 /NAS1

#sudo apt-get install curlftpfs
#curlftpfs -o allow_other test_user:1q2w3e4r5t@intflow.serveftp.com:14147/DL_data /DL_data
#curlftpfs -o allow_other test_user:1q2w3e4r5t@intflow.serveftp.com:14147/NAS1 /NAS1

