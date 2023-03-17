from argparse import ArgumentParser
import os
from os.path import isfile, join
import cv2
import numpy as np
from tqdm import tqdm
from src.models.detection.apas_yolox_allclass import AllClassAPASYOLOXDetModel
from src.models.detection.apas_yolox_2hands import TwoHandAPASYOLOXDetModel
from src.models.detection.base import DetectionModel
from src.models.detection_enhancer.base import DetectionEnhancer
from src.models.detection_enhancer.raw import RawDetectionEnhancer




def get_det_model(name) -> DetectionModel:
    if name == 'apas_yolox_2hands':
        return TwoHandAPASYOLOXDetModel()
    if name == 'apas_yolox_allclass':
        return AllClassAPASYOLOXDetModel()
    raise "unknown det model"

def get_det_enhancer(name) -> DetectionEnhancer:
    if name == 'raw':
        return RawDetectionEnhancer()
    raise "unknown pose enhancer model"

def process_video(det_model:DetectionModel, detection_enhancer:DetectionEnhancer, video_path:str, out_root:str, visualize=False):
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

    detection_features = []
    count = 0
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        
        count += 1
        # if count >= 1200:
        #     break
        
        res, frame = det_model.detect_one_per_category(img, visualize=visualize, thresh=0.5)
        enhanced_detection = detection_enhancer.forward(res)
        detection_features.append(enhanced_detection)
        
        if visualize:
            videoWriter.write(frame)
            
    np.save(os.path.join(out_root,
                         f'{os.path.basename(video_path)}.bbox.npy'),
                np.stack(detection_features))
    
    cap.release()
    if visualize:
        videoWriter.release()
        

def export(det_model, det_enhancer, clip_root, out_root, visualize):
    detection_model: DetectionModel = get_det_model(det_model)
    detectin_enhancer: DetectionEnhancer = get_det_enhancer(det_enhancer)
    
    all_clips = [join(clip_root, f) for f in os.listdir(clip_root) if isfile(join(clip_root, f))]
    for clip_path in tqdm(all_clips):
        process_video(detection_model, detectin_enhancer, clip_path, out_root, visualize)
        # break

def main():
    """[summary]
    This script takes as input the name of a pose estimation model and the name of a pose enhancing model, and 
    performs the pose extraction on all clips in a folder
    """
    parser = ArgumentParser()
    parser.add_argument('det_model', help='Name of detection model')
    parser.add_argument('det_enhancer', help='Name of pose enhancement model')
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