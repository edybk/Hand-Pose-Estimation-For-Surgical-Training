
import itertools
import statistics 
from collections import defaultdict
import os
from typing import Optional
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import json
from scipy import signal
from src.dataset.proxy import Proxy
from mmcv.visualization.image import imshow_det_bboxes
from mmpose.apis import vis_pose_result
import scipy.stats as stats


# num_hands = 4


# def load_raw_poses(poses_path):
#     vid_poses = np.load(poses_path)
#     vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
#     vid_keypoints = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
#     vid_bboxes = np.reshape(vid_bboxes, (-1, num_hands, 5))
#     return vid_keypoints, vid_bboxes

# def get_poses_for_frame(vid_keypoints, vid_bboxes, frame_id, bbox_threshold=0):
#     pose_results = []
#     for i in range(num_hands):
#         if vid_bboxes[frame_id][i][4] > bbox_threshold:
#             pose_results.append({
#                 'keypoints': vid_keypoints[frame_id][i],
#                 'bbox': vid_bboxes[frame_id][i]
#             })
#     return pose_results

# def get_poses_for_video(vid_keypoints, vid_bboxes, bbox_threshold=0):
#     frame_poses = [get_poses_for_frame(vid_keypoints, vid_bboxes, i, bbox_threshold) 
#                     for i in range(len(vid_bboxes))]
#     return [(p['bbox'], p['keypoints']) for p in itertools.chain.from_iterable(frame_poses)]

# def get_keypoints(root):
#     for fname in os.listdir(root):
#         if not fname.endswith(".npy"):
#             continue
#         vid_keypoints, vid_bboxes = load_raw_poses(os.path.join(root, fname))
#         return get_poses_for_video(vid_keypoints, vid_bboxes, bbox_threshold=0)

# def get_keypoints_normalization_params(keypoints_root_path, bb_threshold=0.5, kp_threshold=0.3):
#     kp_norm_vals = defaultdict(lambda: defaultdict(list))
#     bbox_norm_vals = defaultdict(list)

#     for bbox, kps in get_keypoints(keypoints_root_path):
#         if bbox[4] < bb_threshold:
#             continue
#         for i in range(4):
#             bbox_norm_vals[i].append(bbox[i])
#         for i, kp in enumerate(kps):
#             if kp[2] < kp_threshold:
#                 continue
#             kp_norm_vals[i]['x'].append(kp[0])
#             kp_norm_vals[i]['y'].append(kp[1])

#     for kpidx in kp_norm_vals.keys():
#         for coordinate, val in kp_norm_vals[kpidx].items():
#             kp_norm_vals[kpidx][coordinate] = {
#                 'mean': statistics.mean(val),
#                 'std': statistics.stdev(val)
#             }
#     for coordinate, val in bbox_norm_vals.items():
#         bbox_norm_vals[coordinate] = {
#             'mean': statistics.mean(val),
#             'std': statistics.stdev(val)
#         }
 
#     return kp_norm_vals, bbox_norm_vals

# def normalize(v, mean, std):
#     return (v-mean) / std

# def normalize_keypoint_features(vid_poses: np.ndarray, kp_normalization_params:dict, bbox_normalization_params:dict) -> np.ndarray:
#     # vid_poses: [272, vid_len]
#     vid_poses = vid_poses.transpose(1, 0) # [vid_len, 272]
#     vid_len = vid_poses.shape[0]
#     vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
#     vid_keypoints = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
#     for frame_kp in vid_keypoints: #[num_hands, 21, 3]
#         for hand_kp in frame_kp: # [21, 3]
#             for i in range(21):
#                 # normalize x
#                 hand_kp[i][0] = normalize(hand_kp[i][0], kp_normalization_params[str(i)]['x']['mean'], kp_normalization_params[str(i)]['x']['std'])
#                 # normalize y
#                 hand_kp[i][1] = normalize(hand_kp[i][1], kp_normalization_params[str(i)]['y']['mean'], kp_normalization_params[str(i)]['y']['std'])
#     flat_vid_keypoints = np.reshape(vid_keypoints, (vid_len, -1))
#     # print(f"flat_vid_keypoints {flat_vid_keypoints.shape}")
#     vid_bboxes = np.reshape(vid_bboxes, (-1, num_hands, 5))
#     for frame_bboxes in vid_bboxes: #[num_hands, 5]
#         for hand_bbox in frame_bboxes: # [5, ]
#             for i in range(4):
#                 hand_bbox[i] = normalize(hand_bbox[i], bbox_normalization_params[str(i)]['mean'], bbox_normalization_params[str(i)]['std'])
#     flat_vid_bboxes = np.reshape(vid_bboxes, (vid_len, -1))
#     # print(f"flat_vid_bboxes {flat_vid_bboxes.shape}")
#     normalized_vid_poses = np.concatenate((flat_vid_keypoints.transpose(), flat_vid_bboxes.transpose())) # [vid_len, 272]
#     assert(normalized_vid_poses.shape == (272, vid_len))
#     return normalized_vid_poses

# def distance_between_points(x1, y1, x2, y2):
#     return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )       

# def add_fingertip_distances_to_poses(vid_poses_original: np.ndarray) -> np.ndarray:
#     # vid_poses_original: [272, vid_len]
#     vid_poses = vid_poses_original.transpose(1, 0) # [vid_len, 272]
#     vid_len = vid_poses.shape[0]
#     vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
#     vid_keypoints_tmp = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
    
    
    
    
#     finger_tips = [(4, 8),
#                     (4, 12),
#                     (4, 16),
#                     (4, 20),
#                     (8, 12),
#                     (8, 16),
#                     (8, 20),
#                     (12, 16),
#                     (12, 20),
#                     (16, 20)]
    
#     vid_fingertip_distances = []
#     for frame_kps in vid_keypoints_tmp:
#         frame_distances = []
#         for hand_kps in frame_kps:
#             hand_distances = []
#             for tip1, tip2 in finger_tips:
#                 dist = distance_between_points(x1=hand_kps[tip1][0], y1=hand_kps[tip1][1], x2=hand_kps[tip2][0], y2=hand_kps[tip2][1])
#                 # dist = math.sqrt( (hand_kps[tip2][0] - hand_kps[tip1][0])**2 + (hand_kps[tip2][1] - hand_kps[tip1][1])**2 )           
#                 hand_distances.append(dist)
#             frame_distances.append(hand_distances)
#         vid_fingertip_distances.append(frame_distances)
#     vid_fingertip_distances = np.array(vid_fingertip_distances)
#     vid_fingertip_distances = np.reshape(vid_fingertip_distances, (vid_len, -1))
#     # vid_fingertip_distances = np.transpose(vid_fingertip_distances)
#     # print(vid_poses_original.shape)
#     # print(vid_fingertip_distances.shape)
#     vid_poses_original = vid_poses_original.transpose()
#     return np.column_stack((vid_poses_original, vid_fingertip_distances)).transpose()
#     # return np.concatenate((vid_poses_original, vid_fingertip_distances), axis=0)
        
# def _get_iou(bb1, bb2):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes.

#     Parameters
#     ----------
#     bb1 : dict
#         Keys: {'x1', 'x2', 'y1', 'y2'}
#         The (x1, y1) position is at the top left corner,
#         the (x2, y2) position is at the bottom right corner
#     bb2 : dict
#         Keys: {'x1', 'x2', 'y1', 'y2'}
#         The (x, y) position is at the top left corner,
#         the (x2, y2) position is at the bottom right corner

#     Returns
#     -------
#     float
#         in [0, 1]
#     """
#     assert bb1['x1'] < bb1['x2']
#     assert bb1['y1'] < bb1['y2']
#     assert bb2['x1'] < bb2['x2']
#     assert bb2['y1'] < bb2['y2']

#     # determine the coordinates of the intersection rectangle
#     x_left = max(bb1['x1'], bb2['x1'])
#     y_top = max(bb1['y1'], bb2['y1'])
#     x_right = min(bb1['x2'], bb2['x2'])
#     y_bottom = min(bb1['y2'], bb2['y2'])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     # compute the area of both AABBs
#     bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
#     bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
#     assert iou >= 0.0
#     assert iou <= 1.0
#     return iou

# def iou(bbox1, bbox2):
#     x1, y1, x2, y2, conf = bbox1
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     bb1=dict(x1=x1,y1=y1,x2=x2,y2=y2)
#     x1, y1, x2, y2, conf = bbox2
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     bb2=dict(x1=x1,y1=y1,x2=x2,y2=y2)
#     return _get_iou(bb1, bb2)

# def calculate_bbox_area(bbox):
#     x1, y1, x2, y2, conf = bbox
#     return abs(x2-x1) * abs(y2-y1)

# def calculate_bbox_aspect_ratio(bbox):
#     x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
#     width = distance_between_points(x1, y1, x1, y2)
#     height = distance_between_points(x1, y2, x2, y2)
#     try:
#         return (width/height)
#     except:
#         return 0

def calculate_bbox_center(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x = x1+(x2-x1)/2
    y = y1+(y2-y1)/2
    return(x,y)

def calculate_bbox_bottom_right_corner(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x2, y2)

def int_calculate_bbox_center(bbox):
    (x,y) = calculate_bbox_center(bbox)
    return (int(x),int(y))

# def _get_2_hands_by_area(frame_bboxes, frame_keypoints):
    
#     bboxes_areas = [calculate_bbox_area(bbx) for bbx in frame_bboxes]
#     bboxes_centers = [calculate_bbox_center(bbx) for bbx in frame_bboxes]
#     # print(bboxes_areas)
#     # print(bboxes_centers)
#     wanted_bboxes_by_area = sorted(bboxes_areas, reverse=True)[:2]
#     wanted_bboxes_by_index = [bboxes_areas.index(area) for area in wanted_bboxes_by_area]
#     bbox1 = frame_bboxes[wanted_bboxes_by_index[0]]
#     bbox2 = frame_bboxes[wanted_bboxes_by_index[1]]
    
#     keypoints1 = frame_keypoints[wanted_bboxes_by_index[0]]
#     keypoints2 = frame_keypoints[wanted_bboxes_by_index[1]]
    
#     if bbox1[0] > bbox2[0]:
#         tmp_bbox = bbox1
#         tmp_keypoints = keypoints1
#         bbox1 = bbox2
#         keypoints1 = keypoints2
#         bbox2 = tmp_bbox
#         keypoints2 = tmp_keypoints
#     frame_bboxes = [bbox1, bbox2]
#     frame_keypoints = [keypoints1, keypoints2]
#     return frame_bboxes, frame_keypoints

# def _make_iou_key(bbox):
#     def getkey(pose):
#         return iou(bbox, pose['bbox'])
#     return getkey

# def _make_center_distance_key(bbox):
#     x, y = calculate_bbox_center(bbox)
    
#     def getkey(pose):
#         px, py = calculate_bbox_center(pose['bbox'])
#         return distance_between_points(x, y, px, py)
#     return getkey

# def _get_2_hands_by_iou(frame_bboxes, frame_keypoints, prev_bboxes):
#     poses = [dict(bbox=frame_bboxes[i], keypoints=frame_keypoints[i]) for i in range(len(frame_bboxes))]
#     first_pose = sorted(poses, key=_make_iou_key(prev_bboxes[0]), reverse=True)[0]
#     second_pose = sorted(poses, key=_make_iou_key(prev_bboxes[1]), reverse=True)[0]
#     new_frame_bboxes = [first_pose['bbox'], second_pose['bbox']]
#     new_frame_keypoints = [first_pose['keypoints'], second_pose['keypoints']]
#     return new_frame_bboxes, new_frame_keypoints
      

# def _get_2_hands_by_center_distance(frame_bboxes, frame_keypoints, prev_bboxes):
#     poses = [dict(bbox=frame_bboxes[i], keypoints=frame_keypoints[i]) for i in range(len(frame_bboxes))]
#     first_pose = sorted(poses, key=_make_center_distance_key(prev_bboxes[0]), reverse=False)[0]
#     second_pose = sorted(poses, key=_make_center_distance_key(prev_bboxes[1]), reverse=False)[0]
#     new_frame_bboxes = [first_pose['bbox'], second_pose['bbox']]
#     new_frame_keypoints = [first_pose['keypoints'], second_pose['keypoints']]
#     return new_frame_bboxes, new_frame_keypoints  

# def keep_2_hands(vid_poses_original: np.ndarray) -> np.ndarray:
#     # vid_poses_original: [272, vid_len]
#     vid_poses = vid_poses_original.transpose(1, 0) # [vid_len, 272]
#     vid_len = vid_poses.shape[0]
#     vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
#     # print(vid_keypoints.shape)
    
#     vid_keypoints_tmp = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
#     vid_bboxes_tmp = np.reshape(vid_bboxes, (-1, num_hands, 5))
#     # print(vid_keypoints_tmp.shape)
#     # print(vid_bboxes_tmp.shape)
#     bboxes_by_frame = []
#     keypoints_by_frame = []
#     prev_bboxes = None
#     for frame_idx in range(vid_poses.shape[0]):
#         frame_bboxes = vid_bboxes_tmp[frame_idx]
#         frame_keypoints = vid_keypoints_tmp[frame_idx]
#         frame_bboxes, frame_keypoints = _get_2_hands_by_area(frame_bboxes, frame_keypoints)
#         if prev_bboxes is not None:
#             frame_bboxes, frame_keypoints = _get_2_hands_by_center_distance(frame_bboxes, frame_keypoints, prev_bboxes)
        
#         bboxes_by_frame.append(frame_bboxes)
#         keypoints_by_frame.append(frame_keypoints)
#         prev_bboxes = frame_bboxes
#         # print(frame_bboxes)
#         # print(frame_keypoints)
        
#         # if frame_idx > 15:        
#         #     break
    
#     vid_keypoints_tmp = np.array(keypoints_by_frame)
#     vid_bboxes_tmp = np.array(bboxes_by_frame)
#     with_confidence = 1
#     if with_confidence == 0:
#         vid_keypoints_tmp = vid_keypoints_tmp[:, :, :, :2]
#         vid_bboxes_tmp = vid_bboxes_tmp[:, :, :4]
    
#     # print(vid_keypoints_tmp.shape)
#     # print(vid_bboxes_tmp.shape)
#     vid_keypoints_tmp = np.reshape(vid_keypoints_tmp, (-1, 2*21*(2+with_confidence)))
#     vid_bboxes_tmp = np.reshape(vid_bboxes_tmp, (-1, 2*(4+with_confidence)))
#     vid_poses_tmp = np.concatenate((vid_keypoints_tmp, vid_bboxes_tmp), axis=1)
#     # print(vid_poses_tmp.shape)
#     return vid_poses_tmp.transpose()


#def calculate_finger_angles(hand_pose):
    # pose is 21x3

def velocity(x1, x2):
    return (x2-x1) / 2

def acceleration(x1, x2, x3):
    return velocity(velocity(x1, x2), velocity(x2, x3))

def pose_velocity(keypoints1:np.ndarray, keypoints2:np.ndarray):
    """_summary_

    Args:
        keypoints1 (np.ndarray): of shape 21, 3
        keypoints2 (np.ndarray): of shape 21, 3

    Returns:
        np.ndarray: of shape 21,2
    """
    x = velocity(keypoints1[:, 0], keypoints2[:, 0])
    x = x.reshape(21, 1)
    # print(x.shape)
    y = velocity(keypoints1[:, 1], keypoints2[:, 1])
    y = y.reshape(21, 1)
    # print(y.shape)
    return np.concatenate((x, y), axis=1)



def combine_poses_with_detections(vid_poses, vid_detections):
    # print(vid_poses.shape)
    # print(vid_detections.shape)
    
    vid_bboxes_without_hands = vid_detections[2*5:, :]
    combined = np.concatenate((vid_poses, vid_bboxes_without_hands), axis=0)
    return combined

def load_raw_combined_from_numpy(combined_numpy):
    combined_numpy = combined_numpy.transpose()
    vid_len = combined_numpy.shape[0]
    vid_hands = combined_numpy[:, :21*3+5+21*3+5]
    left_keypoints = vid_hands[:, :21*3]
    left_keypoints = left_keypoints.reshape((-1, 21, 3))
    left_bbox = vid_hands[:, 21*3:21*3+5]
    left_bbox = left_bbox.reshape((vid_len, 5))
    right_keypoints = vid_hands[:, 21*3+5:21*3+5+21*3]
    right_keypoints = right_keypoints.reshape((-1, 21, 3))
    right_bbox = vid_hands[:, -5:]
    right_bbox = right_bbox.reshape((vid_len, 5))
    vid_tools = combined_numpy[:, 21*3+5+21*3+5:]
    vid_tools = vid_tools.reshape((vid_len, -1, 5))
    # print(vid_dets.shape)
    return left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools

def load_raw_combined(poses_path):
    combined = np.load(poses_path)
    return load_raw_combined_from_numpy(combined_numpy=combined)

def convert_combined_poses_and_detections_to_centers(combined_features):

    def get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id):
        return left_keypoints[frame_id], left_bbox[frame_id], right_keypoints[frame_id], right_bbox[frame_id], vid_tools[frame_id]

    def convert_frame_features(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools):
        
        left_center = calculate_bbox_center(left_bbox)
        right_center = calculate_bbox_center(right_bbox)
        tools_centers = []
        for bbox in vid_tools:
            tools_centers.append(list(calculate_bbox_center(bbox)))
        combined_centers_for_frame = np.concatenate((left_keypoints[:, :2].flatten(), 
                                                    np.array(list(left_center)).flatten(), 
                                                    right_keypoints[:, :2].flatten(),
                                                    np.array(list(right_center)).flatten(), 
                                                    np.array(tools_centers).flatten()))
        return combined_centers_for_frame
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined_from_numpy(combined_features)
    converted = []
    for frame_id in range(combined_features.shape[1]):
        _left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools = get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id)
        converted.append(convert_frame_features(_left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools))
    converted = np.array(converted)
    return converted.transpose()


def convert_combined_poses_and_detections_to_centers_and_boundaries(combined_features):
    
    def load_raw_combined(combined):
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

    def convert_frame_features(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools):
        # print(left_bbox.shape)
        # raise ""
        left_center = calculate_bbox_center(left_bbox)
        right_center = calculate_bbox_center(right_bbox)
        tools_centers = []
        for bbox in vid_tools:
            tools_centers.append(list(calculate_bbox_center(bbox)))
        combined_centers_for_frame = np.concatenate((left_keypoints[:, :2].flatten(), 
                                                    np.array(list(left_center)).flatten(), 
                                                    left_bbox[:4].flatten(),
                                                    
                                                    right_keypoints[:, :2].flatten(),
                                                    np.array(list(right_center)).flatten(), 
                                                    right_bbox[:4].flatten(),
                                                    
                                                    np.array(tools_centers).flatten(),
                                                    np.array(vid_tools[:, :4]).flatten()))
        return combined_centers_for_frame
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined(combined_features)
    converted = []
    for frame_id in range(combined_features.shape[1]):
        _left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools = get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id)
        converted.append(convert_frame_features(_left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools))
    converted = np.array(converted)
    return converted.transpose()




def convert_combined_poses_and_detections_to_centers_and_thresholds(combined_features):
    
    def load_raw_combined(combined):
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

    def convert_frame_features(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools):
        # print(left_bbox.shape)
        # raise ""
        left_center = calculate_bbox_center(left_bbox)
        left_threshold = left_bbox[4:]
        right_center = calculate_bbox_center(right_bbox)
        right_threshold = right_bbox[4:]
        tools_centers = []
        for bbox in vid_tools:
            tools_centers.append(list(calculate_bbox_center(bbox)))
        combined_centers_for_frame = np.concatenate((left_keypoints[:, :3].flatten(), 
                                                    np.array(list(left_center)).flatten(), 
                                                    np.array(list(left_threshold)).flatten(),
                                                    
                                                    right_keypoints[:, :2].flatten(),
                                                    np.array(list(right_center)).flatten(), 
                                                    np.array(list(right_threshold)).flatten(),
                                                    
                                                    np.array(tools_centers).flatten(),
                                                    np.array(vid_tools[:, 4]).flatten()))
        return combined_centers_for_frame
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined(combined_features)
    converted = []
    for frame_id in range(combined_features.shape[1]):
        _left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools = get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id)
        converted.append(convert_frame_features(_left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools))
    converted = np.array(converted)
    return converted.transpose()



def convert_combined_poses_and_detections_to_centers_and_boundaries_and_thresholds(combined_features):
    
    def load_raw_combined(combined):
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

    def convert_frame_features(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools):
        # print(left_bbox.shape)
        # raise ""
        left_center = calculate_bbox_center(left_bbox)
        right_center = calculate_bbox_center(right_bbox)
        tools_centers = []
        for bbox in vid_tools:
            tools_centers.append(list(calculate_bbox_center(bbox)))
        combined_centers_for_frame = np.concatenate((left_keypoints[:, :3].flatten(), 
                                                    np.array(list(left_center)).flatten(), 
                                                    left_bbox[:4].flatten(),
                                                    np.array(list(left_bbox[4:])).flatten(),
                                                    
                                                    right_keypoints[:, :2].flatten(),
                                                    np.array(list(right_center)).flatten(), 
                                                    right_bbox[:4].flatten(),
                                                    np.array(list(right_bbox[4:])).flatten(),
                                                    
                                                    np.array(tools_centers).flatten(),
                                                    np.array(vid_tools[:, :4]).flatten(),
                                                    np.array(vid_tools[:, 4]).flatten()))
        return combined_centers_for_frame
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined(combined_features)
    converted = []
    for frame_id in range(combined_features.shape[1]):
        _left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools = get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id)
        converted.append(convert_frame_features(_left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools))
    converted = np.array(converted)
    return converted.transpose()



#################### jupyter notebook utils ################


def savgol_filter(keypoints):
    keypoints = keypoints.copy()
    window_length, polyorder = 13, 2
    
    for i in range(keypoints.shape[1]): 
        keypoints[:, i, 0] = signal.savgol_filter(keypoints[:, i, 0], window_length, polyorder)
        keypoints[:, i, 1] = signal.savgol_filter(keypoints[:, i, 1], window_length, polyorder)
    return keypoints



############# surgical skill assessment ######################

def download_all_features_with_bboxes_centers():
    #TODO:
    raise "not implemented"

def load_all_samples(combined_centers_path):
    all_combined = [os.path.join(combined_centers_path, file) for file in os.listdir(combined_centers_path) if file.endswith(".npy")]
    return all_combined

def load_raw_all_combined(raw_combined_path):   
    raw_all_combined = [os.path.join(raw_combined_path, file) for file in os.listdir(raw_combined_path) if file.endswith(".npy")]
    return raw_all_combined

# def vector_from_points(x1,y1,x2,y2):
#     distance = [x2 - x1, y2 - y1]
#     norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
#     direction = [distance[0] / norm, distance[1] / norm]
#     bullet_vector = [direction[0] * math.sqrt(2), direction[1] * math.sqrt(2)]
#     return bullet_vector

def load_sample(all_combined, raw_all_combined, sample_name):
    sample = [x for x in all_combined if sample_name in x][0]
    raw_sample = [x for x in raw_all_combined if sample_name in x][0]
    return raw_sample, sample



def impute_coordinates(vals, mode="last"):
    new_vals = vals.copy()
    # array.shape == -1, 2
    for i in range(1, new_vals.shape[0]):
        if new_vals[i, 0] == 0 and new_vals[i, 1] == 0:
            if mode == 'last':
                new_vals[i, 0] = new_vals[i-1, 0]
                new_vals[i, 1] = new_vals[i-1, 1]
            elif mode == 'interpolate':
                j = i
                while j < new_vals.shape[0] and new_vals[j, 0] == 0 and new_vals[j, 1] == 0:
                    j += 1
                if j == new_vals.shape[0]:
                    new_vals[i:j, 0] = new_vals[i-1, 0]
                    new_vals[i:j, 1] = new_vals[i-1, 1]
                    break
                else:
                    x_vals = np.linspace(
                        start=new_vals[i-1, 0], stop=new_vals[j, 0], num=j-(i-1)+1, endpoint=False)
                    new_vals[i-1:j+1, 0] = x_vals
                    y_vals = np.linspace(
                        start=new_vals[i-1, 1], stop=new_vals[j, 1], num=j-(i-1)+1, endpoint=False)
                    new_vals[i-1:j+1, 1] = y_vals
            else:
                raise "ERROR"
    # special case for first frame
    if new_vals[0, 0] == 0 and new_vals[0, 1] == 0:
        new_vals[0, 0], new_vals[0, 1] = new_vals[1, 0], new_vals[1, 1]
    return new_vals


def impute_keypoints(kps, mode):
    new_kps = kps.copy()
    for i in range(new_kps.shape[1]):
        new_kps[:, i, :] = impute_coordinates(kps[:, i, :], mode)
    return new_kps


def impute_tools(tools, mode):
    new_tools = tools.copy()
    for i in range(tools.shape[1]):
        new_tools[:, i, :] = impute_coordinates(tools[:, i, :], mode)
    return new_tools


def savgol_filter(vals1d):
    # vals = vals1d.copy()
    window_length, polyorder = 13, 2
    window_length, polyorder = 99, 2
    window_length, polyorder = 21, 2
    return signal.savgol_filter(vals1d, window_length, polyorder)


def gaussian_filter(vals1d):
    return gaussian_filter1d(vals1d, sigma=10)


def lowpass_filter(vals1d):
    sos = signal.butter(5, 10, 'hp', fs=30, output='sos')
    sos = signal.butter(1, 2, 'hp', fs=30, output='sos')
    sos = signal.butter(0, 14, 'hp', fs=30, output='sos')
    return signal.sosfilt(sos, vals1d)

def filter_coordinates(vals2d, f):
    vals = vals2d.copy()
    vals[:, 0] = f(vals2d[:, 0])
    vals[:, 1] = f(vals2d[:, 1])
    return vals


def filter_keypoints(kps, mode):
    new_kps = kps.copy()
    for i in range(new_kps.shape[1]):
        new_kps[:, i, :] = filter_coordinates(kps[:, i, :], mode)
    return new_kps


def filter_tools(tools, mode):
    new_tools = tools.copy()
    for i in range(tools.shape[1]):
        new_tools[:, i, :] = filter_coordinates(tools[:, i, :], mode)
    return new_tools

def filter_gestures(gestures, mode):
    new_gestures = gestures.copy()
    new_gestures = filter_coordinates(gestures, mode)
    return new_gestures
    
def get_video_from_poses(videos_dir, poses_file):
    return videos_dir + "/" + os.path.basename(poses_file).split(".")[0] + ".wmv"

def load_tcn_labels(labels_root, vid_name, actions_dict):
    all_labels = [
        f"{labels_root}/{x}" for x in os.listdir(labels_root) if x.endswith(".txt")]
    vid_labels = [x for x in all_labels if vid_name in x][0]
    with open(vid_labels, 'r') as file_ptr:
        content = file_ptr.read().split('\n')[:-1]
    length = len(content)
    classes = np.zeros(length)
    for i in range(len(classes)):
        classes[i] = actions_dict[content[i]]
    return classes


def load_from_raw(poses_path):
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined(
        poses_path)

    assert(left_bbox.shape[0] == vid_tools.shape[0])
    left_centers = []
    right_centers = []
    vid_tools_centers = []

    for i in range(left_bbox.shape[0]):
        left_center = calculate_bbox_center(left_bbox[i])
        left_centers.append(left_center)

        right_center = calculate_bbox_center(right_bbox[i])
        right_centers.append(right_center)

        frame_tools = vid_tools[i]
        frame_centers = []
        for tool_bbox in frame_tools:
            tool_center = calculate_bbox_center(tool_bbox)
            frame_centers.append(tool_center)
        vid_tools_centers.append(frame_centers)

    left_centers = np.array(left_centers)
    right_centers = np.array(right_centers)
    vid_tools_centers = np.array(vid_tools_centers)
    return left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, left_centers, right_centers, vid_tools_centers

def vid_labels_into_clips(vid_labels, actions_dict, gesture_to_name):
    assert(len(vid_labels) != 0)
    reverse_actions_dict = dict([(v, k) for (k, v) in actions_dict.items()])
    clips = []
    current_start = 0
    last_label = vid_labels[0]
    for i, label, in enumerate(vid_labels):
        if label == last_label:
            continue
        else:
            clips.append({
                "start": current_start,
                "end": i-1,
                "label": gesture_to_name[reverse_actions_dict[int(last_label)]]
            })
            current_start = i
            last_label = label
    clips.append({
                "start": current_start,
                "end": len(vid_labels)-1,
                "label": gesture_to_name[reverse_actions_dict[int(last_label)]]
            })
    
    for clip in clips:
        clip_length = clip['end'] - clip['start'] + 1
        clip["length"] = clip_length
        
    return clips

def load_actions_dict(p):
    with open(p, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict

def load_action_to_name(g2n_path):
    with open(g2n_path) as f:
        gesture_to_name = json.load(f)
        return gesture_to_name


def load_action_labels_and_action_dicts(dataset_root_dir):
    gestures_actions_dict = load_actions_dict(os.path.join(dataset_root_dir, "mapping.txt"))
    gestures_labels_root = os.path.join(dataset_root_dir, "groundTruth")

    tool_usage_actions_dict = load_actions_dict(os.path.join(dataset_root_dir, "mapping_tools.txt"))
    left_tool_usage_labels_root = os.path.join(dataset_root_dir, "groundTruthToolsLeft")
    right_tool_usage_labels_root = os.path.join(dataset_root_dir, "groundTruthToolsRight")

    g2n_path = os.path.join(dataset_root_dir, "gesture_to_name.json")
    t2n_path = os.path.join(dataset_root_dir, "tool_to_name.json")

    gestures_action_to_name = load_action_to_name(g2n_path)
    tool_usage_action_to_name = load_action_to_name(t2n_path)

    return gestures_actions_dict, gestures_labels_root, tool_usage_actions_dict, left_tool_usage_labels_root, right_tool_usage_labels_root, gestures_action_to_name, tool_usage_action_to_name

def clip(clip_description, vals):
    clip_start = clip_description['start']
    return vals[clip_start:clip_start+clip_description["length"]] # return vals[clip_start:clip_start+clip_description["length"]+1]

def init_draw_proxy(length):
    vid_proxy_vals = [0] * length
    histories = defaultdict(list)
    return vid_proxy_vals, histories

def visualize_export(video_file, vleft_keypoints, vleft_center, vleft_bboxes, vright_keypoints, vright_center, vright_bboxes, vvid_tools_centers, vvid_tools_bboxes, tag="", gesture_package={}, proxy_package={}, demo=False, clip_description={}, out_root=None, stopat=None):

    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_num = 0
    
    if proxy_package:
        proxy_instance:Proxy = proxy_package["proxy"]
        vid_proxy_vals, histories = init_draw_proxy(vleft_keypoints.shape[0])
        if proxy_package.get("save", False):
            out_video_proxy_path = os.path.join(out_root,
                                        f'{os.path.basename(video_file)}._{tag}.proxy.vis.npy')
    
    if clip_description:
        clip_counter = 0
        
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        
        if stopat and frame_num > stopat:
            break
        
        if clip_description:
            if clip_counter < clip_description["start"]:
                clip_counter += 1
                continue
            if frame_num >= clip_description["length"]:
                break
        
        if frame_num > 100 and demo:
            break

        if frame_num % 500 == 0:
            print(100*(frame_num/frame_count))

        first_frame = img

        frame_fleft_keypoints = vleft_keypoints[frame_num]
        frame_fleft_bboxes = vleft_bboxes[frame_num]
        frame_fright_keypoints = vright_keypoints[frame_num]
        frame_fright_bboxes = vright_bboxes[frame_num]
        frame_fvid_tools_bbox = vvid_tools_bboxes[frame_num]
        frame_fleft_centers = vleft_center[frame_num]
        frame_fright_centers = vright_center[frame_num]
        frame_fvid_tools_centers = vvid_tools_centers[frame_num]
        
        if demo and frame_num == 99:
            cv2.imwrite('visualize_proxy.png', first_frame)
            
        if proxy_package:
            proxy_params = dict(
                 left_keypoints=frame_fleft_keypoints, 
                 left_center=frame_fleft_centers, 
                 left_bboxes=frame_fleft_bboxes,
                 right_keypoints=frame_fright_keypoints, 
                 right_center=frame_fright_centers, 
                 right_bboxes=frame_fright_bboxes, 
                 vid_tools_centers=frame_fvid_tools_centers,
                 vid_tools_bboxes=frame_fvid_tools_bbox,
                 histories=histories)            
            proxy_instance.calculate(proxy_params, frame_num, vid_proxy_vals)
            

        frame_num += 1

    if proxy_package:
        if proxy_package.get("save", False):
            np.save(out_video_proxy_path, np.array(vid_proxy_vals))
        return vid_proxy_vals
    
def get_filter_method(name):
    if name == "savgol":
        the_filter = savgol_filter
    elif name == "gaussian":
        the_filter = gaussian_filter
    elif name == "lowpass":
        the_filter = lowpass_filter
    return the_filter

def end2end_clips_export(dataset_root_dir, videos_dir, all_combined, raw_all_combined, sample_name, kp_clean_method, kp_filter_method, tool_clean_method, tool_filter_method, proxy:Proxy, out_dir=None, output_format:Optional[str]=None, skip_no_gesture = True, clip_by="gestures"):
    raw_sample, sample = load_sample(all_combined, raw_all_combined, sample_name)
    
    sample_base_name = os.path.basename(sample_name).split(".")[0]
    
    video_file = get_video_from_poses(videos_dir, sample)
    print(video_file)

    e2e_left_keypoints, e2e_left_bboxes, e2e_right_keypoints, e2e_right_bboxes, e2e_vid_tools_bbox, e2e_left_centers, e2e_right_centers, e2e_vid_tools_centers = load_from_raw(
        raw_sample)
    
    if kp_clean_method == "last" or kp_clean_method == "interpolate":
        e2e_left_centers = impute_coordinates(e2e_left_centers, mode=kp_clean_method)
        e2e_right_centers = impute_coordinates(e2e_right_centers, mode=kp_clean_method)

        e2e_left_keypoints = impute_keypoints(e2e_left_keypoints,  mode=kp_clean_method)
        e2e_right_keypoints = impute_keypoints(e2e_right_keypoints,  mode=kp_clean_method)
    
    if tool_clean_method == "last" or kp_clean_method == "interpolate":
        e2e_vid_tools_centers = impute_tools(e2e_vid_tools_centers, mode=kp_clean_method)
    

        
    if kp_filter_method:
        the_filter = get_filter_method(kp_filter_method)
        e2e_left_centers = filter_coordinates(e2e_left_centers, the_filter)
        e2e_right_centers = filter_coordinates(e2e_right_centers, the_filter)

        e2e_left_keypoints = filter_keypoints(e2e_left_keypoints,  the_filter)
        e2e_right_keypoints = filter_keypoints(e2e_right_keypoints,  the_filter)
    
    if tool_filter_method:
        the_filter = get_filter_method(tool_filter_method)
        e2e_vid_tools_centers = filter_tools(e2e_vid_tools_centers, the_filter)
    
    out_root = out_dir if out_dir else "./vis_results/visualize_proxies/"
    out_root = f"{out_root}/{sample_base_name}/"


    os.makedirs(out_root, exist_ok=True)
    
    gestures_actions_dict, gestures_labels_root, tool_usage_actions_dict, left_tool_usage_labels_root, right_tool_usage_labels_root, gestures_action_to_name, tool_usage_action_to_name = load_action_labels_and_action_dicts(dataset_root_dir)

    if clip_by == "gestures":
        gestures_labels = load_tcn_labels(gestures_labels_root, sample_name, gestures_actions_dict)
        labels = gestures_labels
        actions_dict = gestures_actions_dict
        action_to_name = gestures_action_to_name
    elif clip_by == "tool_usage_left":
        left_tool_usage_labels = load_tcn_labels(left_tool_usage_labels_root, sample_name, tool_usage_actions_dict)
        labels = left_tool_usage_labels
        actions_dict = tool_usage_actions_dict
        action_to_name = tool_usage_action_to_name
    elif clip_by == "tool_usage_right":
        right_tool_usage_labels = load_tcn_labels(right_tool_usage_labels_root, sample_name, tool_usage_actions_dict)
        labels = right_tool_usage_labels
        actions_dict = tool_usage_actions_dict
        action_to_name = tool_usage_action_to_name
    
    clips = vid_labels_into_clips(labels, actions_dict, action_to_name)
    proxy_package = None
    proxy_tag = ""
    if proxy:
        proxy_package=dict(proxy=proxy, position=2, save=True)
        proxy_tag = "_" + proxy.name()

    def get_tag():
        tg= f"{kp_clean_method}_{kp_filter_method}_{tool_clean_method}_{tool_filter_method}" + proxy_tag
        
        if output_format:
            tg = tg + "_" + str(output_format)
        
        if clip_by != "gestures":
            tg = tg + "_" + str(clip_by)
                
        return tg
    
    # print(gesture_clips)
    all_labels = sorted(list(set([gc["label"] for gc in clips])))

    # print(all_labels)
    
    for label in all_labels:
        if skip_no_gesture and label == "No Gesture":
            continue
        for i, label_clip_description in tqdm(enumerate([x for x in clips if x['label'] == label][:1111111])):

            print(label_clip_description)
            proxy_vals = visualize_export(video_file,
                    clip(label_clip_description, e2e_left_keypoints),
                    clip(label_clip_description, e2e_left_centers),
                    clip(label_clip_description, e2e_left_bboxes),
                    clip(label_clip_description, e2e_right_keypoints),
                    clip(label_clip_description, e2e_right_centers),
                    clip(label_clip_description, e2e_right_bboxes),
                    clip(label_clip_description, e2e_vid_tools_centers),
                    clip(label_clip_description, e2e_vid_tools_bbox),
                    tag=f"{get_tag()}_{label.replace(' ', '_')}_{i}",
                    gesture_package=dict(actions_dict=actions_dict, gt_results=clip(label_clip_description, labels), gesture_to_name=action_to_name, position=1),
                    proxy_package=proxy_package,
                    demo=False,
                    clip_description=label_clip_description,
                    out_root=out_root)
            
                # break
        # break
  
def init_draw_action(actions_dict, gt_results):
    actions_dict['G-1'] = -1
    reverse_actions_dict = dict([(v, k) for k, v in actions_dict.items()])

    ground_truth = ['G-1'] * len(gt_results)

    return reverse_actions_dict, ground_truth


def draw_center(frame, xy, name="", show_label=False):
    int_xy = (int(xy[0]), int(xy[1]))
    frame = cv2.circle(frame, center=int_xy, radius=1,
                       color=(0, 0, 255), thickness=3)
    if show_label:
        frame = cv2.putText(frame, text=name, org=int_xy, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5, color=(255, 0, 0), thickness=1)

    return frame


def draw_bbox(img, bbox, label):
    if len(bbox) == 5:
        bbox = bbox[:4]
    return imshow_det_bboxes(
        img=img,
        bboxes=np.array([bbox]),
        labels=np.array([0]),
        class_names=[label],
        show=False
    )


def draw_bbox_and_center(img, xy, bbox, label):
    img = draw_center(img, xy, name=label, show_label=False)
    img = draw_bbox(img, bbox, label)
    return img


def load_tool_categories(dataset_root):
    detection_class_names_path = os.path.join(dataset_root, "class_names.json")
    with open(detection_class_names_path) as f:
        categories = json.load(f)
    return categories[2:]

s_pose_model = None
def get_pose_model():
    global s_pose_model
    if not s_pose_model:
        from src.models.detection.apas_yolox_allclass import AllClassAPASYOLOXDetModel
        from src.models.pose.resnet import TwoHandsResnetPoseModel
        pose_model = TwoHandsResnetPoseModel(AllClassAPASYOLOXDetModel(), left_cat=2, right_cat=1).pose_model
        s_pose_model = pose_model
    return s_pose_model

def draw_poses(img, poses, show=False):
    res = vis_pose_result(
        get_pose_model(),
        img,
        poses,
        dataset='APASDataset',
        dataset_info=None,
        kpt_score_thr=0,
        radius=4,  # radius,
        thickness=1,  # thickness,
        show=False)
    if show:
        cv2.imwrite('visualize_poses.png', res)
    return res



FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1
label_color = (255, 255, 255)
msg_color = (128, 128, 128)


def _white_to_transparency(img):
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img




def _draw_prediction_bars(ground_truth, actions_dict):
    # draw gestures as time series
    cmap_ = "tab10"
    vmax_ = len(actions_dict)
    ground_truth

    fig, axs = plt.subplots(1, 1, figsize=(14,  0.7*1))
    # axs.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fig.suptitle(f'{""}', fontsize=1)
    gestures_ = []
    if "numpy.ndarray" in str(type(ground_truth)):
        ground_truth = ground_truth.tolist()
    for gest in ground_truth:
        gestures_.append(actions_dict[gest])
    ground_truth = np.array(gestures_)
    map = np.tile(ground_truth, (100, 1))
    axs.axis('off')
    # axs[i].set_title(names[i], fontsize=10, pad= 10)
    axs.imshow(map, cmap=cmap_, aspect="auto",
               interpolation='nearest', vmax=vmax_)

    plt.tight_layout()
    plt.close(fig)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img



def draw_prediction_bars_on_frame(frame, ground_truth, actions_dict, position=1):
    img = _draw_prediction_bars(ground_truth, actions_dict)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_frame = pil_frame.convert('RGBA')
    width, height = pil_frame.size
    prediction_bar = Image.fromarray(img)
    prediction_bar = prediction_bar.resize((int(0.7*width), int(0.1*height)))
    prediction_bar = _white_to_transparency(prediction_bar)
    pil_frame.paste(prediction_bar, (int(0.15*width),
                    height-int(position*int(0.1*height))), prediction_bar)
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def draw_gesture(frame, i, ground_truth_array, actions_dict, reverse_actions_dict, gt_results, gesture_to_name, position=1):
    try:
        ground_truth_array[i] = reverse_actions_dict[gt_results[i].item()]
        frame = draw_prediction_bars_on_frame(
            frame, ground_truth_array, actions_dict, position)
        location = (0, 40+int(20*position))
        text = f"Actual: {gesture_to_name[reverse_actions_dict[int(gt_results[i].item())]]}"
        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                    label_color, THICKNESS, LINETYPE)
    except Exception as e:
        print(e)
    return frame

def visualize(dataset_root, video_file, vleft_keypoints, vleft_center, vleft_bboxes, vright_keypoints, vright_center, vright_bboxes, vvid_tools_centers, vvid_tools_bboxes, tag="", gesture_package={}, proxy_package={}, demo=False, clip_description={}, out_root=None, stopat=None):
    print(stopat)
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video_path = os.path.join(out_root,
                                  f'{os.path.basename(video_file)}._{tag}.vis.wmv')
    print(out_video_path)
    out_video_blank_path = os.path.join(out_root,
                                        f'{os.path.basename(video_file)}._{tag}.blank.vis.wmv')

    print(out_video_path)
    
    if not demo:
        videoWriter = cv2.VideoWriter(
            out_video_path, fourcc,
            fps, size)

        videoWriterBlank = cv2.VideoWriter(
            out_video_blank_path, fourcc,
            fps, size)

    frame_num = 0

    if gesture_package:
        if type(gesture_package) is list:
            gesture_init_vars = [init_draw_action(gp["actions_dict"], gp["gt_results"])
                                                  for gp in gesture_package]
        else:
            reverse_actions_dict, ground_truth = init_draw_action(
                gesture_package["actions_dict"], gesture_package["gt_results"])

    if proxy_package:
        proxy_instance:Proxy = proxy_package["proxy"]
        vid_proxy_vals, histories = init_draw_proxy(vleft_keypoints.shape[0])
        if proxy_package.get("save", False):
            out_video_proxy_path = os.path.join(out_root,
                                        f'{os.path.basename(video_file)}._{tag}.proxy.vis.npy')
    
    if clip_description:
        clip_counter = 0
        
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        # print(frame_num)

        
        if clip_description:
            if clip_counter < clip_description["start"]:
                clip_counter += 1
                continue
            if frame_num >= clip_description["length"]:
                break

        if frame_num % 500 == 0:
            print(100*(frame_num/frame_count))

        first_frame = img
        # print(first_frame.shape)
        blank_frame = np.zeros(first_frame.shape, np.uint8)

        frame_fleft_keypoints = vleft_keypoints[frame_num]
        frame_fleft_bboxes = vleft_bboxes[frame_num]
        frame_fright_keypoints = vright_keypoints[frame_num]
        frame_fright_bboxes = vright_bboxes[frame_num]
        frame_fvid_tools_bbox = vvid_tools_bboxes[frame_num]
        frame_fleft_centers = vleft_center[frame_num]
        frame_fright_centers = vright_center[frame_num]
        frame_fvid_tools_centers = vvid_tools_centers[frame_num]

        if not demo or (demo and ((not stopat and frame_num == 399) or (stopat and frame_num == stopat) or frame_num == frame_count-10)):

            # draw hands bbox and center
            # left
            first_frame = draw_bbox_and_center(
                first_frame, xy=frame_fleft_centers, bbox=frame_fleft_bboxes, label="Left Hand")
            blank_frame = draw_bbox_and_center(
                blank_frame, xy=frame_fleft_centers, bbox=frame_fleft_bboxes, label="Left Hand")
            # right
            first_frame = draw_bbox_and_center(
                first_frame, xy=frame_fright_centers, bbox=frame_fright_bboxes, label="Right Hand")
            blank_frame = draw_bbox_and_center(
                blank_frame, xy=frame_fright_centers, bbox=frame_fright_bboxes, label="Right Hand")

            # draw tools bbox and center
            categories = load_tool_categories(dataset_root)
            for bbox, center, category in zip(frame_fvid_tools_bbox, frame_fvid_tools_centers, categories):
                label = category["name"]
                first_frame = draw_bbox_and_center(
                    first_frame, xy=center, bbox=bbox, label=label)
                blank_frame = draw_bbox_and_center(
                    blank_frame, xy=center, bbox=bbox, label=label)

            # draw hands keypoints
            first_frame = draw_poses(first_frame, [{'keypoints': frame_fleft_keypoints, 'bbox': frame_fleft_bboxes}, {
                                    'keypoints': frame_fright_keypoints, 'bbox': frame_fright_bboxes}], show=False)
            blank_frame = draw_poses(blank_frame, [{'keypoints': frame_fleft_keypoints, 'bbox': frame_fleft_bboxes}, {
                                    'keypoints': frame_fright_keypoints, 'bbox': frame_fright_bboxes}], show=False)

            if gesture_package:
                if type(gesture_package) is list:
                    for gp, (gp_reverse_actions_dict, gp_ground_truth) in zip(gesture_package, gesture_init_vars):
                        first_frame = draw_gesture(first_frame, frame_num, gp_ground_truth,
                                                gp["actions_dict"], gp_reverse_actions_dict, gp["gt_results"], gp["gesture_to_name"], position=gp.get("position", 1))
                        blank_frame = draw_gesture(blank_frame, frame_num, gp_ground_truth,
                                                gp["actions_dict"], gp_reverse_actions_dict, gp["gt_results"], gp["gesture_to_name"], position=gp.get("position", 1))

                else:
                    first_frame = draw_gesture(first_frame, frame_num, ground_truth,
                                            gesture_package["actions_dict"], reverse_actions_dict, gesture_package["gt_results"], gesture_package["gesture_to_name"], position=gesture_package.get("position", 1))
                    blank_frame = draw_gesture(blank_frame, frame_num, ground_truth,
                                            gesture_package["actions_dict"], reverse_actions_dict, gesture_package["gt_results"], gesture_package["gesture_to_name"], position=gesture_package.get("position", 1))

            
        if proxy_package:
            proxy_params = dict(
                 left_keypoints=frame_fleft_keypoints, 
                 left_center=frame_fleft_centers, 
                 left_bboxes=frame_fleft_bboxes,
                 right_keypoints=frame_fright_keypoints, 
                 right_center=frame_fright_centers, 
                 right_bboxes=frame_fright_bboxes, 
                 vid_tools_centers=frame_fvid_tools_centers,
                 vid_tools_bboxes=frame_fvid_tools_bbox,
                 histories=histories)
            
            first_frame = proxy_instance.draw(first_frame, proxy_params, vid_proxy_vals, frame_num, position=proxy_package.get("position", 1))
            blank_frame = proxy_instance.draw(blank_frame, proxy_params, vid_proxy_vals, frame_num, position=proxy_package.get("position", 1))
        
        if demo:
            # print(f"stopat={stopat}, frame_num={frame_num}, frame_count={frame_count}")
            if (not stopat and frame_num == 399) or (stopat and frame_num == stopat) or frame_num == frame_count-10:
                cv2.imwrite(f'./vis_results/visualize_proxy_{proxy_instance.name()}.png', first_frame)
                cv2.imwrite(f'./vis_results/visualize_proxy_{proxy_instance.name()}_blank.png', blank_frame)
                break
        else:
            videoWriter.write(first_frame)
            videoWriterBlank.write(blank_frame)

        frame_num += 1

    if not demo:
        videoWriter.release()
        videoWriterBlank.release()

    if proxy_package:
        if proxy_package.get("save", False):
            np.save(out_video_proxy_path, np.array(vid_proxy_vals))
        return vid_proxy_vals
 


def end2end(dataset_root_dir, videos_dir, all_combined, raw_all_combined, sample_name, kp_clean_method, kp_filter_method, tool_clean_method, tool_filter_method, proxy:Proxy, test=None, out_root_dir=None):

    
    raw_sample, sample = load_sample(all_combined, raw_all_combined, sample_name)    
    
    video_file = get_video_from_poses(videos_dir, sample)
    print(video_file)

    e2e_left_keypoints, e2e_left_bboxes, e2e_right_keypoints, e2e_right_bboxes, e2e_vid_tools_bbox, e2e_left_centers, e2e_right_centers, e2e_vid_tools_centers = load_from_raw(
        raw_sample)
    
    if kp_clean_method == "last" or kp_clean_method == "interpolate":
        e2e_left_centers = impute_coordinates(e2e_left_centers, mode=kp_clean_method)
        e2e_right_centers = impute_coordinates(e2e_right_centers, mode=kp_clean_method)

        e2e_left_keypoints = impute_keypoints(e2e_left_keypoints,  mode=kp_clean_method)
        e2e_right_keypoints = impute_keypoints(e2e_right_keypoints,  mode=kp_clean_method)
    
    if tool_clean_method == "last" or kp_clean_method == "interpolate":
        e2e_vid_tools_centers = impute_tools(e2e_vid_tools_centers, mode=kp_clean_method)
    
    
        
    if kp_filter_method:
        the_filter = get_filter_method(kp_filter_method)
        e2e_left_centers = filter_coordinates(e2e_left_centers, the_filter)
        e2e_right_centers = filter_coordinates(e2e_right_centers, the_filter)

        e2e_left_keypoints = filter_keypoints(e2e_left_keypoints,  the_filter)
        e2e_right_keypoints = filter_keypoints(e2e_right_keypoints,  the_filter)
    
    if tool_filter_method:
        the_filter = get_filter_method(tool_filter_method)
        e2e_vid_tools_centers = filter_tools(e2e_vid_tools_centers, the_filter)
    
    out_root = out_root_dir or "./vis_results/visualize_proxies/"

    should_save = test is None
    if should_save:
        os.makedirs(out_root, exist_ok=True)
    
    gestures_actions_dict, gestures_labels_root, tool_usage_actions_dict, left_tool_usage_labels_root, right_tool_usage_labels_root, gestures_action_to_name, tool_usage_action_to_name = load_action_labels_and_action_dicts(dataset_root_dir)

    gestures_labels = load_tcn_labels(gestures_labels_root, sample_name, gestures_actions_dict)
    # left_tool_usage_labels = load_tcn_labels(left_tool_usage_labels_root, sample_name, tool_usage_actions_dict)
    # right_tool_usage_labels = load_tcn_labels(right_tool_usage_labels_root, sample_name, tool_usage_actions_dict)

    def get_tag():
        return f"{kp_clean_method}_{kp_filter_method}_{tool_clean_method}_{tool_filter_method}"
    
    proxy_package = None
    if proxy:
        proxy_package=dict(proxy=proxy, position=2, save=should_save)
    # print(video_file)
    visualize(dataset_root_dir,
              video_file,
              e2e_left_keypoints,
              e2e_left_centers,
              e2e_left_bboxes,
              e2e_right_keypoints,
              e2e_right_centers,
              e2e_right_bboxes,
              e2e_vid_tools_centers,
              e2e_vid_tools_bbox,
              tag=get_tag(),
              gesture_package=dict(actions_dict=gestures_actions_dict, gt_results=gestures_labels, gesture_to_name=gestures_action_to_name, position=1),
              proxy_package=proxy_package,
              out_root=out_root,
              demo=not should_save,
              stopat=test)




def all_gestures(dataset_dir):
    with open(os.path.join(dataset_dir, "gesture_to_name.json")) as f:
        return [v.replace(" ", "_") for v in json.load(f).values()]
    
def all_tools(dataset_dir):
    with open(os.path.join(dataset_dir, "tool_to_name.json")) as f:
        return [v.replace(" ", "_") for v in json.load(f).values()]



def get_all_novice_samples():
    novices = [
        "P020",
        "P022",
        "P023",
        "P024",
        "P026",
        "P027",
        "P028",
        "P029",
        "P030",
        # "P031", # this one is left handed!
        "P033"
        ]
    actions = ['tissue1', 'tissue2', 'balloon1', 'balloon2']
    vids = []
    for s in novices:
        for a in actions:
            vids.append(s + '_' + a)
    return vids



def get_all_expert_samples():
    experts = ["P016",
                "P018",
                "P019",
                "P021",
                "P025",
                "P032",
                "P034",
                "P036",
                "P037",
                "P038",
                "P039",
                "P040"]
    actions = ['tissue1', 'tissue2', 'balloon1', 'balloon2']
    vids = []
    for s in experts:
        for a in actions:
            vids.append(s + '_' + a)
    return vids

def get_excluded_expert_samples():
    experts = ["P035"]
    actions = ['tissue1', 'tissue2', 'balloon1', 'balloon2']
    vids = []
    for s in experts:
        for a in actions:
            vids.append(s + '_' + a)
    return vids 

def load_proxy(vid_dir, label):
    clips = sorted([f"{vid_dir}/{x}" for x in os.listdir(vid_dir) if label in x])
    valvals = []
    for c in clips:
        if not c.endswith(".npy"):
            continue
        proxy_vals = np.load(c)
        valvals.append(proxy_vals)
    return valvals

def p_value(group1, group2):
    #perform two sample t-test with equal variances
    statistic, pvalue = stats.ttest_ind(a=group1, b=group2, equal_var=True)
    # print(f"t test = {statistic} p value = {pvalue}")
    return pvalue


def get_proxy_mean_p_value(label, all_vid_dirs1, all_vid_dirs2):

    def get_proxy_means(vid_dirs):
        final_means = []
        for vid in vid_dirs:
            the_vals = load_proxy(vid ,label) # this is an array of arrays where each array has the metric for all frames
            the_means = [np.mean(np.array(x)) for x in the_vals] # this is an array of means for the different novice runs on this gesture
            mean = np.mean(the_means) # this is the average value of the novice's metric average over the different runs
            final_means.append(mean)
        return final_means

    means1 = get_proxy_means(all_vid_dirs1)
    means2 = get_proxy_means(all_vid_dirs2)
    return p_value(means1, means2)


def aggregate_proxy_mean_of_means(all_vid_dirs, label, aggregate=np.mean):
    final_mean = []
    for vid in all_vid_dirs:
        the_vals = load_proxy(vid ,label) # this is an array of arrays where each array has the metric for all frames
        the_means = [aggregate(np.array(x)) for x in the_vals] # this is an array of means for the different novice runs on this gesture
        mean = np.mean(the_means) # this is the average value of the novice's metric average over the different runs
        final_mean.append(mean)
    len_before = len(final_mean)
    final_mean = [x for x in final_mean if not np.isnan(x)]
    len_after = len(final_mean)
    if len_before != len_after:
        print(f"removed {len_before-len_after} vids due to nan")
    final_mean = np.mean(final_mean)
    return final_mean

def plot_bars(labels,
              proxy,
              export_dir,
              save=False,
              aggregate=np.mean,
              max_p_value=None):
    
    save_target = os.path.join("vis_results", "proxy_plots")
    if save:
        print(f"Saving figures to {save_target}")
        os.makedirs(save_target, exist_ok=True)

    all_novice_samples = get_all_novice_samples()
    
    all_novice_vid_dirs = [f"{export_dir}/{proxy}/{novice_s}" for novice_s in all_novice_samples]
    
    all_expert_samples = get_all_expert_samples() 
    
    all_expert_vid_dirs = [f"{export_dir}/{proxy}/{expert_s}" for expert_s in all_expert_samples]
    
    fig = plt.figure(figsize = (10, 5))
    
    plt.title(proxy)
    
    plt.xlabel("Gesture")
    plt.ylabel("Proxy Mean")

    
    label_to_vals = defaultdict(list)
    labels = [lbl for lbl in labels if "gesture" not in lbl.lower()]
    skip_plot = True if max_p_value is not None else False
    for label in labels:
        if max_p_value:
            pval = get_proxy_mean_p_value(label, all_novice_vid_dirs, all_expert_vid_dirs)
            if pval <= max_p_value:
                print(f"plotting {label} for proxy {proxy} because p value = {pval}")
                skip_plot = False
            else:
                print(f"skipping {label} for proxy {proxy} because p value = {pval}")
        final_novice_mean = aggregate_proxy_mean_of_means(all_novice_vid_dirs, label, aggregate=aggregate)
        final_expert_mean = aggregate_proxy_mean_of_means(all_expert_vid_dirs, label, aggregate=aggregate)
        
        label_to_vals[label, 'novice'] = final_novice_mean 
        label_to_vals[label, 'expert'] = final_expert_mean
    
    if skip_plot:
        return

    br1 = np.arange(len(labels))
    br2 = [x + 0.25 for x in br1]
    plt.xticks([r + 0.25 for r in range(len(labels))],
        labels)
    
    novice_vals = [label_to_vals[label, 'novice'] for label in labels]
    expert_vals = [label_to_vals[label, 'expert'] for label in labels]
    
    plt.bar(br1, novice_vals, color ='r', width = 0.25,
            edgecolor ='grey', label ='Novice')
    plt.bar(br2, expert_vals, color ='g', width = 0.25,
            edgecolor ='grey', label ='Expert')
    
    # add values on top
    max_val = max(max(novice_vals), max(expert_vals))
    up_delta = 0.05 * max_val
    for i, v in enumerate(novice_vals):
        plt.text(br1[i]-0.09, max(0, min(max_val, v - up_delta)), "{:.2f}".format(v))
        
    for i, v in enumerate(expert_vals):
        plt.text(br2[i]-0.09, max(0, min(max_val, v - up_delta)), "{:.2f}".format(v))
    # end
    
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_target, f"bar_chart_{proxy}.png"))
    
    # plt.show()
    

def aggregate_means(all_vid_dirs, label):
    final_means = []
    for vid in all_vid_dirs:
        the_vals = load_proxy(vid ,label) # this is an array of arrays where each array has the metric for all frames
        the_means = [np.mean(np.array(x)) for x in the_vals] # this is an array of means for the different student runs on this gesture
        mean = np.mean(the_means) # this is the average value of the student's metric average over the different runs
        final_means.append(mean)
    return final_means

def plot_box_plots(labels,
              proxy,
              export_dir,
              save=False):
    save_target = os.path.join("vis_results", "proxy_plots")
    if save:
        print(f"Saving figures to {save_target}")
        os.makedirs(save_target, exist_ok=True)

    all_novice_samples = get_all_novice_samples() # ["P022_balloon2"] #
    
    all_novice_vid_dirs = [f"{export_dir}/{proxy}/{novice_s}" for novice_s in all_novice_samples]
    
    all_expert_samples = get_all_expert_samples() # ["P040_balloon2"] #
    
    all_expert_vid_dirs = [f"{export_dir}/{proxy}/{expert_s}" for expert_s in all_expert_samples]
    
    
    label_to_vals = defaultdict(list)
    labels = [lbl for lbl in labels if "gesture" not in lbl.lower()]
    for label in labels:
        final_novice_mean = aggregate_means(all_novice_vid_dirs, label)
        final_expert_mean = aggregate_means(all_expert_vid_dirs, label)
        
        label_to_vals[label, 'novice'] = final_novice_mean 
        label_to_vals[label, 'expert'] = final_expert_mean
    
    novice_vals = [label_to_vals[label, 'novice'] for label in labels]
    expert_vals = [label_to_vals[label, 'expert'] for label in labels]
    
    
    def plot_3_by_3_novice_vs_expert():
        fig, axes = plt.subplots(nrows=math.ceil(len(labels)/2.0), ncols=2, figsize=(15, 10))

        j = 0
        for i, (label,sval,mvals) in enumerate(zip(labels, novice_vals, expert_vals)):
            
            # fig = plt.figure(figsize = (10, 5))
            plt.subplot(math.ceil(len(labels)/2.0), 2, i+1)
            
            axes[int(i/2)][j].boxplot([sval, mvals], labels=['Novice', 'Expert'])
            j = (j+1) % 2
            plt.title(f"{proxy}")
                
            plt.xlabel(f"{label}")
            plt.ylabel("Proxy Mean")
        
        if len(labels) % 2 != 0:
            # center last plot
            last_i = math.ceil(len(labels)/2.0)-1
            last_j = 1
            axes[last_i][last_j].set_visible(False)
        
        fig.tight_layout()
        
        # plt.legend()
        if save:
            plt.savefig(os.path.join(save_target, f"box_plot_{proxy}.png"))
        
        # plt.show()
        
    def plot_1_by_1_novice_then_expert():
        for some_vals, t in [(novice_vals, 'novice'), (expert_vals, 'expert')]:
            fig = plt.figure(figsize = (10, 5))
            
            plt.title(f"{t}_{proxy}")
            
            plt.xlabel("Gesture")
            plt.ylabel("Proxy Mean")
            
            br1 = np.arange(len(labels))
            # br2 = [x + 1 for x in br1]
            
            plt.boxplot(some_vals, labels=labels, positions=br1)
            
        
            # plt.legend()
            if save:
                plt.savefig(os.path.join(save_target, f"1_1_box_plot_{proxy}.png"))
            
            # plt.show()
    
    
    plot_3_by_3_novice_vs_expert()
    # plot_1_by_1_novice_then_expert()
  