#!/bin/bash

# install

nvcc -V
gcc --version
python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
rm -rf mmdetection
git clone https://github.com/edybk/mmdetection.git
cd mmdetection
python -m pip install -e .

# check installation
python ../scripts/validate_installation.py
