
import os
import json
from scipy import signal

from tqdm import tqdm
from src.dataset.utils import int_calculate_bbox_center, calculate_bbox_center
from  mmcv.visualization.image import imshow_det_bboxes
import numpy as np
import cv2
from mmpose.apis import (vis_pose_result)
from PIL import Image
import matplotlib.pyplot as plt
# from src.models.detection.apas_cascadercnn_finetuned import APASCascadeRCNNDetModel
from src.models.detection.apas_yolox_allclass import AllClassAPASYOLOXDetModel
from src.models.pose.resnet import TwoHandsResnetPoseModel
pose_model = TwoHandsResnetPoseModel(AllClassAPASYOLOXDetModel(), left_cat=2, right_cat=1).pose_model

categories = [
                {
                    "supercategory": "Defect",
                    "id": 1,
                    "name": "Right_hand"
                },
                {
                    "supercategory": "Defect",
                    "id": 2,
                    "name": "Left_hand"
                },
                {
                    "supercategory": "Defect",
                    "id": 3,
                    "name": "Needle_driver"
                },
                {
                    "supercategory": "Defect",
                    "id": 4,
                    "name": "Forceps"
                },
                {
                    "supercategory": "Defect",
                    "id": 5,
                    "name": "Forceps_not_used"
                },
                {
                    "supercategory": "Defect",
                    "id": 6,
                    "name": "Scissors"
                }
            ]

def cv2_imshow(img):
    # load image using cv2....and do processing.
    obj = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()
    return obj

def get_frame(video_name, frame_no):
    cap = cv2.VideoCapture(video_name) #video_name is the video being called
    cap.set(1,frame_no); # Where frame_no is the frame you want
    ret, frame = cap.read() # Read the frame
    return frame
    # cv2.imshow('window_name', frame) # show frame on window
    
def get_video_from_poses(videos_dir, poses_file):
    return videos_dir + "/" + os.path.basename(poses_file).split(".")[0] + ".wmv"

def draw_bbox(img, bbox, label):
    return imshow_det_bboxes(
            img=img,
            bboxes=np.array([bbox]),
            labels=np.array([0]),
            class_names=[label],
            show=False
        )
    

def draw_poses(img, poses, show=True):
    fig = vis_pose_result(
        pose_model,
        img,
        poses,
        dataset='APASDataset',
        dataset_info=None,
        # kpt_score_thr=0.3,#kpt_thr,
        kpt_score_thr=0.1,#kpt_thr,
        radius=4,#radius,
        thickness=1,#thickness,
        show=False)
    if show:
        cv2_imshow(fig)

    return fig
    
def load_raw_combined(poses_path):
    combined = np.load(poses_path)
    combined = combined.transpose()
    vid_len = combined.shape[0]
    vid_hands = combined[:, :21*3+5+21*3+5]
    left_keypoints = vid_hands[:, :21*3]
    left_keypoints = left_keypoints.reshape((-1, 21, 3))
    left_bbox = vid_hands[:, 21*3:21*3+5]
    left_bbox = left_bbox.reshape((vid_len, 5))
    right_keypoints = vid_hands[:, 21*3+5:21*3+5+21*3]
    right_keypoints = right_keypoints.reshape((-1, 21, 3))
    right_bbox = vid_hands[:, -5:]
    right_bbox = right_bbox.reshape((vid_len, 5))
    vid_tools = combined[:, 21*3+5+21*3+5:]
    vid_tools = vid_tools.reshape((vid_len, -1, 5))
    # print(vid_dets.shape)
    return left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools

def get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id):
    return left_keypoints[frame_id], left_bbox[frame_id], right_keypoints[frame_id], right_bbox[frame_id], vid_tools[frame_id]



def savgol_filter(keypoints):
    keypoints = keypoints.copy()
    # window_length, polyorder = 13, 2
    window_length, polyorder = 21, 2
    
    for i in range(keypoints.shape[1]): 
        keypoints[:, i, 0] = signal.savgol_filter(keypoints[:, i, 0], window_length, polyorder)
        keypoints[:, i, 1] = signal.savgol_filter(keypoints[:, i, 1], window_length, polyorder)
    return keypoints

def clean_keypoints(keypoints):
    keypoints = keypoints.copy()
    for fi, fkp in enumerate(keypoints):
        for ii, ikp in enumerate(fkp):
            for i in range(2):
                if ikp[i] == 0 and fi > 0:
                    ikp[i] = keypoints[fi-1, ii, i]
    return keypoints

def postprocess_dataset(videos_dir, combined_path, out_root, visualize):
    all_combined = [os.path.join(combined_path, file) for file in os.listdir(combined_path) if file.endswith(".npy")]
    os.makedirs(out_root, exist_ok=True)
    print(f"num combined: {len(all_combined)}")
    for annotation_file in tqdm(all_combined):
        left_keypointssss, left_bboxsss, right_keypointssss, right_bboxsss, vid_toolssss = load_raw_combined(annotation_file)
        
        left_keypointssss = savgol_filter(clean_keypoints(left_keypointssss))
        right_keypointssss = savgol_filter(clean_keypoints(right_keypointssss))
        
        # print(f"left_keypoints shape: {left_keypointssss.shape}")
        # print(f"left_bbox shape: {left_bboxsss.shape}")
        # print(f"right_keypoints shape: {right_keypointssss.shape}")
        # print(f"right_bbox shape: {right_bboxsss.shape}")
        # print(f"vid_tools shape: {vid_toolssss.shape}")
        video_file = get_video_from_poses(videos_dir, annotation_file)
        print(video_file)
        cap = cv2.VideoCapture(video_file)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(out_root,
                            f'{os.path.basename(video_file)}.smooth.vis.wmv')
        out_export_path = os.path.join(out_root,
                            f'{os.path.basename(video_file.replace(".wmv", ""))}.npy')
        print(out_video_path)

        out_video_blank_path = os.path.join(out_root,
                            f'{os.path.basename(video_file)}.smooth.blank.vis.wmv')
        
        if visualize:
            videoWriter = cv2.VideoWriter(
                out_video_path, fourcc,
                fps, size)
            
            videoWriterBlank = cv2.VideoWriter(
                out_video_blank_path, fourcc,
                fps, size)
        total_res = []
        frame_num = 0

        while (cap.isOpened()):
            flag, img = cap.read()
            if not flag:
                break
            first_frame = img
            # print(first_frame.shape)
            if visualize:
                blank_frame = np.zeros(first_frame.shape, np.uint8)


            left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = get_combined_for_frame(left_keypointssss, left_bboxsss, right_keypointssss, right_bboxsss, vid_toolssss, frame_num)
            # myobj = draw_bboxes(first_frame, pose_results)
            if visualize:
                first_frame = draw_bbox(first_frame, left_bbox, "left hand")
                blank_frame = draw_bbox(blank_frame, left_bbox, "left hand")
            
            left_center = int_calculate_bbox_center(left_bbox)
            if visualize:
                first_frame = cv2.circle(first_frame, center=left_center, radius=1, color=(0, 0, 255), thickness=3)
                blank_frame = cv2.circle(blank_frame, center=left_center, radius=1, color=(0, 0, 255), thickness=3)
            # left_center = calculate_bbox_center(left_bbox)
            
            if visualize:
                first_frame = draw_bbox(first_frame, right_bbox, "right hand")
                blank_frame = draw_bbox(blank_frame, right_bbox, "right hand")

            right_center = int_calculate_bbox_center(right_bbox)
            if visualize:
                first_frame = cv2.circle(first_frame, center=right_center, radius=1, color=(0, 0, 255), thickness=3)
                blank_frame = cv2.circle(blank_frame, center=right_center, radius=1, color=(0, 0, 255), thickness=3)
            # right_center = calculate_bbox_center(right_bbox)

            tools_centers = []
            for bbox, category in zip(vid_tools, categories[2:]):
                if visualize:
                    label = category["name"]
                    # print(f"label {label} bbox {bbox}")
                    first_frame = draw_bbox(first_frame, bbox, label)
                    blank_frame = draw_bbox(blank_frame, bbox, label)
                    center = int_calculate_bbox_center(bbox)
                    first_frame = cv2.circle(first_frame, center=center, radius=1, color=(0, 0, 255), thickness=3)
                    blank_frame = cv2.circle(blank_frame, center=center, radius=1, color=(0, 0, 255), thickness=3)
                tools_centers.append(list(calculate_bbox_center(bbox)))
            # cv2_imshow(first_frame)
            combined_centers_for_frame = np.concatenate((left_keypoints[:, :2].flatten(), 
                                                        np.array(list(left_center)).flatten(), 
                                                        right_keypoints[:, :2].flatten(),
                                                        np.array(list(right_center)).flatten(), 
                                                        np.array(tools_centers).flatten()))
            total_res.append(combined_centers_for_frame)
            # print(combined_centers_for_frame.shape)
            if visualize:
                first_frame = draw_poses(first_frame, [{'keypoints':left_keypoints, 'bbox': left_bbox}, {'keypoints': right_keypoints, 'bbox': right_bbox}], show=False)
                blank_frame = draw_poses(blank_frame, [{'keypoints':left_keypoints, 'bbox': left_bbox}, {'keypoints': right_keypoints, 'bbox': right_bbox}], show=False)
                
                videoWriter.write(first_frame)
                videoWriterBlank.write(blank_frame)
            frame_num += 1

            
            
        
        proxy_vals = np.array(total_res)
        np.save(out_export_path, proxy_vals)
        if visualize:
            videoWriter.release()
            videoWriterBlank.release()

        
        # break