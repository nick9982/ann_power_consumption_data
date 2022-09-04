#!/bin/bash

sudo apt-get update

sudo apt install libjsoncpp-dev -y

cmake -B build && cd build/src && make && ./nn
