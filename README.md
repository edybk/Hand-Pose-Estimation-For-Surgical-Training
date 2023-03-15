# Using Hand Pose Estimation To Automate Surgical Training Feedback

This is the official implementation of the paper [Using Hand Pose Estimation To Automate Surgical Training Feedback](http://google.com).
The repository contains code to reproduce the following experiments:
- Train an object detection model on the Open Surgery Simulation Dataset
- Train an pose estimation model on the Open Surgery Simulation Dataset
- Use the trained models to generate the full pose dataset
- Train a multi-task action segmentation model on I3D and pose inputs
- Visualize the results 
- Calculate surgical skill proxies based on the poses and detections
