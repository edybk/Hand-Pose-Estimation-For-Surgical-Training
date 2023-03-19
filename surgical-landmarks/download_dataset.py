import os
import argparse
from src.scripts.download_features import prepare_features
from src.scripts.download_apas_tcn_v2 import prepare_dataset

repository_root = os.getenv("REPOSITORY_ROOT")

parser = argparse.ArgumentParser(
                    prog='Dataset Downloader',
                    description='Downloads the Open Surgery Simulation dataset detections and pose estimation features as well as I3D features for both frontal and closeup views.',
                    epilog='')

parser.add_argument('--download_location', required = False, type=str, default=os.path.join(repository_root, "surgical-landmarks", "data", "downloaded"))

args = parser.parse_args()

assert args.download_location != ''

data_root = os.path.join(repository_root, "surgical-landmarks", "data")
prepare_dataset(data_root, None)
prepare_features(args.download_location, data_root)