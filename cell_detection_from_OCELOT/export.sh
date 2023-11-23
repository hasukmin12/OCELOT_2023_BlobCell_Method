#!/usr/bin/env bash

./build.sh

# Change the name of your docker algo. or compressed algo.
docker save ocelot23algo | gzip -c > hasukmin_Focal_3k.tar.gz
