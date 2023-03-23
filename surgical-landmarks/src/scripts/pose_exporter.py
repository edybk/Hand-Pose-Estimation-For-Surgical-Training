from argparse import ArgumentParser
import os
from os.path import isfile, join
import cv2
import numpy as np
from tqdm import tqdm

from src.models.detection.apas_yolox_2hands import TwoHandAPASYOLOXDetModel
from src.models.detection.apas_yolox_allclass import AllClassAPASYOLOXDetModel

from src.models.detection.base import DetectionModel

from src.models.pose.base import TwoHandPoseModel
from src.models.pose.resnet import TwoHandsResnetPoseModel
from src.models.pose_enhancer.base import TwoHandPoseEnhancer
from src.models.pose_enhancer.raw import TwoHandRawPoseEnhancer


def get_det_model(name) -> DetectionModel:
    if name == 'apas_yolox_2hands':
        left_hand_cat_id = 1
        right_hand_cat_id = 2
        return TwoHandAPASYOLOXDetModel(), left_hand_cat_id, right_hand_cat_id
    if name == 'apas_yolox_allclass':
        left_hand_cat_id = 2
        right_hand_cat_id = 1
        return AllClassAPASYOLOXDetModel(detection_categories=[0, 1]), left_hand_cat_id, right_hand_cat_id
    raise "unknown det model"

def get_pose_model(detection_model, left_hand_cat_id, right_hand_cat_id, name) -> TwoHandPoseModel:
    if name == 'resnet':
        return TwoHandsResnetPoseModel(detection_model, left_cat=left_hand_cat_id, right_cat=right_hand_cat_id)
    raise "unknown pose model"

def get_pose_enhancer(name) -> TwoHandPoseEnhancer:
    if name == 'raw':
        return TwoHandRawPoseEnhancer()
    raise "unknown pose enhancer model"

def process_video(pose_model:TwoHandPoseModel, pose_enhancer:TwoHandPoseEnhancer, video_path:str, out_root:str, visualize=False):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'
    
    os.makedirs(out_root, exist_ok=True)
    
    if visualize:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(out_root,
                         f'{os.path.basename(video_path)}.vis.wmv'), fourcc,
            fps, size)

    pose_embeddings = []
    count = 0
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        
        count += 1
        # if count >= 120:
        #     break
        
        res = pose_model.forward(img, visualize=visualize)
        enhanced_poses = pose_enhancer.forward(res)
        pose_embeddings.append(enhanced_poses)
        
        if visualize:
            videoWriter.write(res.img)
            
    np.save(os.path.join(out_root,
                         f'{os.path.basename(video_path)}.kpt.npy'),
                np.stack(pose_embeddings))
    
    cap.release()
    if visualize:
        videoWriter.release()
        

def export(det_model, pose_model, pose_enhancer, clip_root, out_root, visualize):
    detection_model, left_hand_cat_id, right_hand_cat_id = get_det_model(det_model)
    pose_model:TwoHandPoseModel = get_pose_model(detection_model, left_hand_cat_id, right_hand_cat_id, pose_model)
    pose_enhancer:TwoHandPoseEnhancer = get_pose_enhancer(pose_enhancer)
    all_clips = [join(clip_root, f) for f in os.listdir(clip_root) if isfile(join(clip_root, f))]
    for clip_path in tqdm(all_clips):
        process_video(pose_model, pose_enhancer, clip_path, out_root, visualize)
        # break

def main():
    """[summary]
    This script takes as input the name of a pose estimation model and the name of a pose enhancing model, and 
    performs the pose extraction on all clips in a folder
    """
    parser = ArgumentParser()
    parser.add_argument('det_model', help='Name of detection model')
    parser.add_argument('pose_model', help='Name of pose estimation model')
    parser.add_argument('pose_enhancer', help='Name of pose enhancement model')
    parser.add_argument('--clip-root', type=str, default='', help='Root directory of clips')
    parser.add_argument('--out-root', type=str, default='', help='Root directory of output')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the output')
    args = parser.parse_args()

    assert args.pose_model != ''
    assert args.pose_enhancer != ''
    assert args.clip_root != ''
    assert args.out_root != ''
    export(args.pose_model, args.pose_enhancer, args.clip_root, args.out_root, args.visualize)
        
        
# if __name__ == 'main':
#     main()