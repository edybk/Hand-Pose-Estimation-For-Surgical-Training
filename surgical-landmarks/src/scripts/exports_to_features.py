import numpy as np
import os
from tqdm import tqdm
def convert(export_dir, features_dir):
    #load poses
    
    all_exports = [os.path.join(export_dir, file) for file in os.listdir(export_dir) if file.endswith(".npy")]

    print(f"num exports: {len(all_exports)}")
    os.makedirs(features_dir, exist_ok=True)
    for ex in tqdm(all_exports):
        feature_file_name = os.path.basename(ex).split(".")[0] + ".npy"
        feature_file_path = os.path.join(features_dir, feature_file_name)
        npex = np.load(ex)
        features = np.transpose(npex)
        np.save(feature_file_path, features)
        