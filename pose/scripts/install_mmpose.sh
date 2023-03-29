#!/bin/bash

# install

nvcc -V
gcc --version

# install dependencies: (use cu111 because colab has CUDA 11.1)
python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
python -m pip install mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

# install mmdet for inference demo
python -m pip install mmdet

# clone mmpose repo
rm -rf mmpose
# !git clone https://github.com/open-mmlab/mmpose.git
git clone https://github.com/edybk/mmpose.git
cd mmpose

# install mmpose dependencies
python -m pip install -r requirements.txt

# install mmpose in develop mode
python -m pip install -e .

# check installation
python ../scripts/validate_installation.py
