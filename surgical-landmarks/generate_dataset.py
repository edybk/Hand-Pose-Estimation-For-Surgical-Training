import os
import argparse
from src.scripts.detections_exporter import export as export_detections
from src.scripts.pose_exporter import export as export_poses
from src.scripts.exports_to_features import convert
from src.scripts.combine_poses_with_detections import combine
from src.scripts.convert_combined_poses_with_detections_to_centers import convert_centers
from src.scripts.export_smoothened import postprocess_dataset

repository_root = os.getenv("REPOSITORY_ROOT")

parser = argparse.ArgumentParser(
                    prog='Dataset Generator',
                    description='Exports bounding box detections and 2D hand poses and optionally draws them on the videos.',
                    epilog='')

parser.add_argument('--detection_model', required = False, type=str, choices = ['apas_yolox_2hands', 'apas_yolox_allclass'], default='apas_yolox_allclass')
parser.add_argument('--pose_model', required = False, type=str, choices = ['resnet'], default='resnet')
parser.add_argument('--videos_root', required = False, type=str, default=os.getenv("FRONTAL_VIDEOS"))
parser.add_argument('--out_root', required = False, type=str, default=os.path.join(repository_root, "surgical-landmarks", "data", "generated"))
parser.add_argument('--draw', required = False, type=bool, default=False)

args = parser.parse_args()

# export detections
detections_out_root = os.path.join(args.out_root, "detection_exports")
export_detections(det_model=args.detection_model,
       det_enhancer='raw', 
       clip_root=args.videos_root, 
       out_root=detections_out_root,
       visualize=args.draw)
detections_features_path = os.path.join(detections_out_root, "features")
convert(export_dir=detections_out_root, 
        features_dir=detections_features_path)

# export poses
pose_out_root = os.path.join(args.out_root, "pose_exports")
export_poses(det_model=args.detection_model,
       pose_model=args.pose_model, 
       pose_enhancer='raw', 
       clip_root=args.videos_root, 
       out_root=pose_out_root,
       visualize=args.draw)
pose_features_path = os.path.join(pose_out_root, "features")
convert(export_dir=pose_out_root, 
        features_dir=pose_features_path)

# combine detections with poses as features
combined_out_root = os.path.join(args.out_root, "poses_detections_combined_features")
combine(pose_features_path, detections_features_path, combined_out_root)

# keep only the centers of the bounding boxes
combined_centers_out_root = os.path.join(args.out_root, "poses_detections_combined_centers_features")
convert_centers(combined_out_root, combined_centers_out_root)

# post-process the dataset to impute missing values and apply smoothing
smooth_out_root = os.path.join(args.out_root, "smooth_final")
postprocess_dataset(
    videos_dir=args.videos_root,
    combined_path=combined_out_root,
    out_root=smooth_out_root,
    visualize=args.draw
)