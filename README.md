# Using Hand Pose Estimation To Automate Open Surgery Training Feedback

This is the official implementation of the paper [Using Hand Pose Estimation To Automate Open Surgery Training Feedback](https://doi.org/10.1007/s11548-023-02947-6).
The repository contains code to reproduce the following experiments:
- [Train an object detection model on the Open Surgery Simulation Dataset](detection/README.md)
- [Train a pose estimation model on the Open Surgery Simulation Dataset](pose/README.md)
- [Use the trained models for multi-task action segmentation and surgical skill assessment](surgical-landmarks/README.md)

  - Generate the full pose dataset
  - Train a multi-task action segmentation model on I3D and pose inputs
  - Visualize the results 
  - Calculate surgical skill proxies based on the poses and detections

The experiments were conducted using python 3.7 and CUDA 11 on a Tesla V100 GPU with 32GB memory.

[Download Dataset](https://forms.gle/TyfiuUHaDu3iJKKr8) | [Scalpel Lab Website](https://scalpel.group)

### Citation
Bkheet, E., D’Angelo, AL., Goldbraikh, A. et al. Using hand pose estimation to automate open surgery training feedback. Int J CARS (2023). https://doi.org/10.1007/s11548-023-02947-6
