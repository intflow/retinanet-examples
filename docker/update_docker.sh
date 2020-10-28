#!/bin/bash

sudo docker commit odtk_melting_cow odtk:melting_cow
sudo docker login docker.io -u kmjeon -p 1011910119a!
sudo docker tag odtk:melting_cow intflow/odtk:melting_cow
sudo docker push intflow/odtk:melting_cow
