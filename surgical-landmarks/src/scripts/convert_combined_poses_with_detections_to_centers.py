import numpy as np
import os
from src.dataset.utils import convert_combined_poses_and_detections_to_centers
from tqdm import tqdm


def convert_centers(combined_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    for features_fname in tqdm(sorted(os.listdir(combined_path))):

        features_fpath = f"{combined_path}/{features_fname}"
        print(features_fpath)
        if not os.path.isfile(features_fpath):
            continue
        combined_features = np.load(features_fpath)

        converted_combined_fpath = f"{output_path}/{features_fname}"
        converted_combined_features = convert_combined_poses_and_detections_to_centers(combined_features)
        # print(converted_combined_features.shape)
        # break
        np.save(converted_combined_fpath, converted_combined_features)
