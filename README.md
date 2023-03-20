# Using Hand Pose Estimation To Automate Surgical Training Feedback

This is the official implementation of the paper [Using Hand Pose Estimation To Automate Surgical Training Feedback](http://google.com).
The repository contains code to reproduce the following experiments:
- [Train an object detection model on the Open Surgery Simulation Dataset](detection/README.md)
- [Train a pose estimation model on the Open Surgery Simulation Dataset](pose/README.md)
- [Use the trained models for multi-task action segmentation and surgical skill assessment](surgical-landmarks/README.md)

  - Generate the full pose dataset
  - Train a multi-task action segmentation model on I3D and pose inputs
  - Visualize the results 
  - Calculate surgical skill proxies based on the poses and detections

The experiments were conducted using python 3.7 and CUDA 11 on a Tesla V100 GPU with 32GB memory.
