# Normalize Keypoints
import json
import numpy as np
import os
from src.dataset.utils import combine_poses_with_detections
from tqdm import tqdm

def combine(poses_path, detections_path, output_path):

    os.makedirs(output_path, exist_ok=True)
    # print(len(all_features))

    for poses_fname, detections_fname in tqdm(zip(
                sorted(os.listdir(poses_path)), 
                sorted(os.listdir(detections_path)
            ))):

        poses_fpath = f"{poses_path}/{poses_fname}"
        detections_fpath = f"{detections_path}/{detections_fname}"
        if not os.path.isfile(poses_fpath) or not os.path.isfile(detections_fpath):
            continue
        # print(poses_fpath)
        # print(detections_fpath)
        combined_fpath = f"{output_path}/{poses_fname}"
        vid_poses = np.load(poses_fpath)
        vid_detections = np.load(detections_fpath)
        
        combined_features = combine_poses_with_detections(vid_poses, vid_detections)
        # print(combined_features.shape)
        # break
        np.save(combined_fpath, combined_features)
