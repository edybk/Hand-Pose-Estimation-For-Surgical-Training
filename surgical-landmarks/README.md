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

    |                 	| Accuracy     	| Edit  	| F1@0.1 	| F1@0.25 	| F1@0.5 	| Command                                                                                                                                                                                                                                                                                                                                                                                                    	|
    |-----------------	|--------------	|-------	|--------	|---------	|--------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
    |                 	|              	|       	|        	|         	|        	|                                                                                                                                                                                                                                                                                                                                                                                                            	|
    | Frontal         	|              	|       	|        	|         	|        	|                                                                                                                                                                                                                                                                                                                                                                                                            	|
    | Keypoints       	| 81.22 ± 5.61 	| 83.61 	| 86.39  	| 82.62   	| 67.41  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 96 --custom-features smooth_final --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1                                                                                                                                                        	|
    | I3D             	| 83.11 ± 5.84 	| 86.35 	| 89.10  	| 86.18   	| 73.76  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 2048 --custom-features i3d_frontal_split --append_split_to_features --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1                                                                                                                      	|
    | I3D + Keypoints 	| 83.24 ± 6.11 	| 85.60 	| 88.33  	| 84.90   	| 71.85  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 2144 --custom-features i3d_frontal_split --append_split_to_features --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final.json --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1         	|
    | Closeup         	|              	|       	|        	|         	|        	|                                                                                                                                                                                                                                                                                                                                                                                                            	|
    | Keypoints       	| 84.16 ± 5.37 	| 79.95 	| 84.25  	| 82.02   	| 72.63  	| python run.py --action train --dataset apas_tcn_v2 --custom-features smooth_final_closeup --features_dim 96 --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --model MHTCN2 --multitask --split all --num_epochs 150 --eval-rate 1                                                                                                                      	|
    | I3D             	| 87.16 ± 4.72 	| 84.66 	| 89.70  	| 88.33   	| 82.10  	| python run.py --action train --dataset apas_tcn_v2 --features_dim 2048 --custom-features i3d_closeup_split --append_split_to_features --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --model MHTCN2 --multitask --split all --num_epochs 150 --eval-rate 1                                                                                                                      	|
    | I3D + Keypoints 	| 87.69 ± 4.40 	| 83.32 	| 88.16  	| 86.72   	| 79.99  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 2144 --custom-features i3d_closeup_split --append_split_to_features --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final_closeup.json --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1 	|
    | Multi-View      	|              	|       	|        	|         	|        	|                                                                                                                                                                                                                                                                                                                                                                                                            	|
    | Keypoints       	| 85.39 ± 4.35 	| 81.63 	| 85.76  	| 84.16   	| 75.83  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 192 --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final_closeup.json --custom-features smooth_final --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1                                  	|
    | I3D             	| 87.89 ± 4.13 	| 85.76 	| 90.61  	| 89.08   	| 82.82  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 4096 --custom-features i3d_both_split --append_split_to_features --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1                                                                                                                         	|
    | I3D + Keypoints 	| 88.35 ± 4.15 	| 85.32 	| 89.68  	| 88.28   	| 82.32  	| python run.py --action train --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 4288 --custom-features i3d_both_split --append_split_to_features --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final_multiview.json --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split all --num_epochs 150 --eval-rate 1  	|

- Testing a multi-task action segmentation model with pretrained weights
    ```
    python run.py --action test --model MHTCN2 --multitask --dataset apas_tcn_v2 --features_dim 4288 --custom-features i3d_both_split --append_split_to_features --appended-features $REPOSITORY_ROOT/surgical-landmarks/data/apas_tcn_v2/appended_features/smooth_final_multiview.json --num_layers_R=10 --num_layers_PG=11 --num_f_maps=64 --num_R=1 --lr=0.001 --split 1 
    ```
- Visualizing the segmentation performance on a sample video

- Calculating surgical skill proxies

