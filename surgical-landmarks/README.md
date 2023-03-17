# Using Hand Pose Estimation To Automate Surgical Training Feedback

Prerequisites:
- Setup [detection](../detection/README.md) with MMDetection and validate the test results
- Setup [pose estimation](../pose/README.md) with MMPose and validate the test results
- Download the videos of Open Surgery Simulation Dataset from [Scalpel's website](https://scalpel.group) and setup the following environment variables:
    - _FRONTAL_VIDEOS_ to the root folder of the videos
        ```
        export FRONTAL_VIDEOS=<path_to_videos>
        ```
    - _REPOSITORY_ROOT_ to the root of this repository
        ```
        export REPOSITORY_ROOT=<path_to_repository_root>
        ```

Reproducing the experiments:
- Setup the environment
    ```
    conda env create -f surgical-landmarks.yml
    conda activate surgical-landmarks
    ```
- Generating the pose dataset based on the trained detection and pose models
    ```
    ./generate_dataset.sh
    ```
- Post-processsing the dataset (includes imputation and smoothing)
    ```
    ./post_process_dataset.sh
    ```
- Training a multi-task action segmentation model

- Visualizing the segmentation performance on a sample video

- Evaluating surgical skill proxies

