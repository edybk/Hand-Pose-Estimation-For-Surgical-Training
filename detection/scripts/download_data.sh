#!/bin/bash

cd mmdetection
mkdir data && cd data
python -m pip install gdown
gdown 1Z2v-ks2DHZ3ubWs36GS-llYL_9WQLWvB
unzip apas_allclass.zip
rm apas_allclass.zip
cd ..