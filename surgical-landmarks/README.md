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
- Option 1: Generating the pose dataset from scratch based on the trained detection and pose models
    ```
    python generate_dataset.py
    ```
    to visualize the detections run with --draw true

- Option 2: Downloading the pre-generated dataset of detections+keypoints and I3D features for both views
    ```
    python download_dataset.py --download_location <download_destination>
    ```

- Training a multi-task action segmentation model

    Example command for training a multi-task action segmentation with frontal keypoints and detections as input:
    ```
    python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 96 --custom-features smooth_final --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1
    ```
    Refer to [TRAIN.md](TRAIN.md) for the full table of model and input options

- Testing a multi-task action segmentation model with pretrained weights

    Example command for testing a multi-task action segmentation with frontal keypoints and detections as input on split number 1:
    ```
    python run.py --action test --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 96 --custom-features smooth_final --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split 1 --ckpt 14RmFXHyYuM-9Xuhh0UI-3radbwn5tNbh
    ```
    Refer to [TEST.md](TEST.md) for the full table of model and input options
- Visualizing the segmentation performance on a sample video
    ```
    python run.py --action visualize --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 4288 --custom-features i3d_both_split1 --append_split_to_features --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final_multiview.json --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split 1 --ckpt "1HTMZHVMATuabr0umGcGdEMlXiPY2YxDW" --vid-name P016_balloon1
    ```
    ![image](https://user-images.githubusercontent.com/12495665/226204079-191114b0-76f0-428b-a822-8afd30951380.png)

- Calculating surgical skill proxies

