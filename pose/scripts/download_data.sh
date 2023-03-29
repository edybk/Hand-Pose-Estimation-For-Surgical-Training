#!/bin/bash

cd mmpose
mkdir data && cd data
python -m pip install gdown
gdown 11FJDkpBM32HRTJo2bAgrg8Oxzi9GQkCc
unzip apas.zip
rm apas.zip
cd ..