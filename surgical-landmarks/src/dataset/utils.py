
import itertools
import statistics 
from collections import defaultdict
import os
import numpy as np
import math
num_hands = 4


def load_raw_poses(poses_path):
    vid_poses = np.load(poses_path)
    vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
    vid_keypoints = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
    vid_bboxes = np.reshape(vid_bboxes, (-1, num_hands, 5))
    return vid_keypoints, vid_bboxes

def get_poses_for_frame(vid_keypoints, vid_bboxes, frame_id, bbox_threshold=0):
    pose_results = []
    for i in range(num_hands):
        if vid_bboxes[frame_id][i][4] > bbox_threshold:
            pose_results.append({
                'keypoints': vid_keypoints[frame_id][i],
                'bbox': vid_bboxes[frame_id][i]
            })
    return pose_results

def get_poses_for_video(vid_keypoints, vid_bboxes, bbox_threshold=0):
    frame_poses = [get_poses_for_frame(vid_keypoints, vid_bboxes, i, bbox_threshold) 
                    for i in range(len(vid_bboxes))]
    return [(p['bbox'], p['keypoints']) for p in itertools.chain.from_iterable(frame_poses)]

def get_keypoints(root):
    for fname in os.listdir(root):
        if not fname.endswith(".npy"):
            continue
        vid_keypoints, vid_bboxes = load_raw_poses(os.path.join(root, fname))
        return get_poses_for_video(vid_keypoints, vid_bboxes, bbox_threshold=0)

def get_keypoints_normalization_params(keypoints_root_path, bb_threshold=0.5, kp_threshold=0.3):
    kp_norm_vals = defaultdict(lambda: defaultdict(list))
    bbox_norm_vals = defaultdict(list)

    for bbox, kps in get_keypoints(keypoints_root_path):
        if bbox[4] < bb_threshold:
            continue
        for i in range(4):
            bbox_norm_vals[i].append(bbox[i])
        for i, kp in enumerate(kps):
            if kp[2] < kp_threshold:
                continue
            kp_norm_vals[i]['x'].append(kp[0])
            kp_norm_vals[i]['y'].append(kp[1])

    for kpidx in kp_norm_vals.keys():
        for coordinate, val in kp_norm_vals[kpidx].items():
            kp_norm_vals[kpidx][coordinate] = {
                'mean': statistics.mean(val),
                'std': statistics.stdev(val)
            }
    for coordinate, val in bbox_norm_vals.items():
        bbox_norm_vals[coordinate] = {
            'mean': statistics.mean(val),
            'std': statistics.stdev(val)
        }
 
    return kp_norm_vals, bbox_norm_vals

def normalize(v, mean, std):
    return (v-mean) / std

def normalize_keypoint_features(vid_poses: np.ndarray, kp_normalization_params:dict, bbox_normalization_params:dict) -> np.ndarray:
    # vid_poses: [272, vid_len]
    vid_poses = vid_poses.transpose(1, 0) # [vid_len, 272]
    vid_len = vid_poses.shape[0]
    vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
    vid_keypoints = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
    for frame_kp in vid_keypoints: #[num_hands, 21, 3]
        for hand_kp in frame_kp: # [21, 3]
            for i in range(21):
                # normalize x
                hand_kp[i][0] = normalize(hand_kp[i][0], kp_normalization_params[str(i)]['x']['mean'], kp_normalization_params[str(i)]['x']['std'])
                # normalize y
                hand_kp[i][1] = normalize(hand_kp[i][1], kp_normalization_params[str(i)]['y']['mean'], kp_normalization_params[str(i)]['y']['std'])
    flat_vid_keypoints = np.reshape(vid_keypoints, (vid_len, -1))
    # print(f"flat_vid_keypoints {flat_vid_keypoints.shape}")
    vid_bboxes = np.reshape(vid_bboxes, (-1, num_hands, 5))
    for frame_bboxes in vid_bboxes: #[num_hands, 5]
        for hand_bbox in frame_bboxes: # [5, ]
            for i in range(4):
                hand_bbox[i] = normalize(hand_bbox[i], bbox_normalization_params[str(i)]['mean'], bbox_normalization_params[str(i)]['std'])
    flat_vid_bboxes = np.reshape(vid_bboxes, (vid_len, -1))
    # print(f"flat_vid_bboxes {flat_vid_bboxes.shape}")
    normalized_vid_poses = np.concatenate((flat_vid_keypoints.transpose(), flat_vid_bboxes.transpose())) # [vid_len, 272]
    assert(normalized_vid_poses.shape == (272, vid_len))
    return normalized_vid_poses

def distance_between_points(x1, y1, x2, y2):
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )       

def add_fingertip_distances_to_poses(vid_poses_original: np.ndarray) -> np.ndarray:
    # vid_poses_original: [272, vid_len]
    vid_poses = vid_poses_original.transpose(1, 0) # [vid_len, 272]
    vid_len = vid_poses.shape[0]
    vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
    vid_keypoints_tmp = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
    
    
    
    
    finger_tips = [(4, 8),
                    (4, 12),
                    (4, 16),
                    (4, 20),
                    (8, 12),
                    (8, 16),
                    (8, 20),
                    (12, 16),
                    (12, 20),
                    (16, 20)]
    
    vid_fingertip_distances = []
    for frame_kps in vid_keypoints_tmp:
        frame_distances = []
        for hand_kps in frame_kps:
            hand_distances = []
            for tip1, tip2 in finger_tips:
                dist = distance_between_points(x1=hand_kps[tip1][0], y1=hand_kps[tip1][1], x2=hand_kps[tip2][0], y2=hand_kps[tip2][1])
                # dist = math.sqrt( (hand_kps[tip2][0] - hand_kps[tip1][0])**2 + (hand_kps[tip2][1] - hand_kps[tip1][1])**2 )           
                hand_distances.append(dist)
            frame_distances.append(hand_distances)
        vid_fingertip_distances.append(frame_distances)
    vid_fingertip_distances = np.array(vid_fingertip_distances)
    vid_fingertip_distances = np.reshape(vid_fingertip_distances, (vid_len, -1))
    # vid_fingertip_distances = np.transpose(vid_fingertip_distances)
    # print(vid_poses_original.shape)
    # print(vid_fingertip_distances.shape)
    vid_poses_original = vid_poses_original.transpose()
    return np.column_stack((vid_poses_original, vid_fingertip_distances)).transpose()
    # return np.concatenate((vid_poses_original, vid_fingertip_distances), axis=0)
        
def _get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def iou(bbox1, bbox2):
    x1, y1, x2, y2, conf = bbox1
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bb1=dict(x1=x1,y1=y1,x2=x2,y2=y2)
    x1, y1, x2, y2, conf = bbox2
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bb2=dict(x1=x1,y1=y1,x2=x2,y2=y2)
    return _get_iou(bb1, bb2)

def calculate_bbox_area(bbox):
    x1, y1, x2, y2, conf = bbox
    return abs(x2-x1) * abs(y2-y1)

def calculate_bbox_center(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x = x1+(x2-x1)/2
    y = y1+(y2-y1)/2
    return(x,y)

def int_calculate_bbox_center(bbox):
    (x,y) = calculate_bbox_center(bbox)
    return (int(x),int(y))

def _get_2_hands_by_area(frame_bboxes, frame_keypoints):
    
    bboxes_areas = [calculate_bbox_area(bbx) for bbx in frame_bboxes]
    bboxes_centers = [calculate_bbox_center(bbx) for bbx in frame_bboxes]
    # print(bboxes_areas)
    # print(bboxes_centers)
    wanted_bboxes_by_area = sorted(bboxes_areas, reverse=True)[:2]
    wanted_bboxes_by_index = [bboxes_areas.index(area) for area in wanted_bboxes_by_area]
    bbox1 = frame_bboxes[wanted_bboxes_by_index[0]]
    bbox2 = frame_bboxes[wanted_bboxes_by_index[1]]
    
    keypoints1 = frame_keypoints[wanted_bboxes_by_index[0]]
    keypoints2 = frame_keypoints[wanted_bboxes_by_index[1]]
    
    if bbox1[0] > bbox2[0]:
        tmp_bbox = bbox1
        tmp_keypoints = keypoints1
        bbox1 = bbox2
        keypoints1 = keypoints2
        bbox2 = tmp_bbox
        keypoints2 = tmp_keypoints
    frame_bboxes = [bbox1, bbox2]
    frame_keypoints = [keypoints1, keypoints2]
    return frame_bboxes, frame_keypoints

def _make_iou_key(bbox):
    def getkey(pose):
        return iou(bbox, pose['bbox'])
    return getkey

def _make_center_distance_key(bbox):
    x, y = calculate_bbox_center(bbox)
    
    def getkey(pose):
        px, py = calculate_bbox_center(pose['bbox'])
        return distance_between_points(x, y, px, py)
    return getkey

def _get_2_hands_by_iou(frame_bboxes, frame_keypoints, prev_bboxes):
    poses = [dict(bbox=frame_bboxes[i], keypoints=frame_keypoints[i]) for i in range(len(frame_bboxes))]
    first_pose = sorted(poses, key=_make_iou_key(prev_bboxes[0]), reverse=True)[0]
    second_pose = sorted(poses, key=_make_iou_key(prev_bboxes[1]), reverse=True)[0]
    new_frame_bboxes = [first_pose['bbox'], second_pose['bbox']]
    new_frame_keypoints = [first_pose['keypoints'], second_pose['keypoints']]
    return new_frame_bboxes, new_frame_keypoints
      

def _get_2_hands_by_center_distance(frame_bboxes, frame_keypoints, prev_bboxes):
    poses = [dict(bbox=frame_bboxes[i], keypoints=frame_keypoints[i]) for i in range(len(frame_bboxes))]
    first_pose = sorted(poses, key=_make_center_distance_key(prev_bboxes[0]), reverse=False)[0]
    second_pose = sorted(poses, key=_make_center_distance_key(prev_bboxes[1]), reverse=False)[0]
    new_frame_bboxes = [first_pose['bbox'], second_pose['bbox']]
    new_frame_keypoints = [first_pose['keypoints'], second_pose['keypoints']]
    return new_frame_bboxes, new_frame_keypoints  

def keep_2_hands(vid_poses_original: np.ndarray) -> np.ndarray:
    # vid_poses_original: [272, vid_len]
    vid_poses = vid_poses_original.transpose(1, 0) # [vid_len, 272]
    vid_len = vid_poses.shape[0]
    vid_keypoints, vid_bboxes = vid_poses[:, :-num_hands*5], vid_poses[:, -num_hands*5:]
    # print(vid_keypoints.shape)
    
    vid_keypoints_tmp = np.reshape(vid_keypoints, (-1, num_hands, 21, 3))
    vid_bboxes_tmp = np.reshape(vid_bboxes, (-1, num_hands, 5))
    # print(vid_keypoints_tmp.shape)
    # print(vid_bboxes_tmp.shape)
    bboxes_by_frame = []
    keypoints_by_frame = []
    prev_bboxes = None
    for frame_idx in range(vid_poses.shape[0]):
        frame_bboxes = vid_bboxes_tmp[frame_idx]
        frame_keypoints = vid_keypoints_tmp[frame_idx]
        frame_bboxes, frame_keypoints = _get_2_hands_by_area(frame_bboxes, frame_keypoints)
        if prev_bboxes is not None:
            frame_bboxes, frame_keypoints = _get_2_hands_by_center_distance(frame_bboxes, frame_keypoints, prev_bboxes)
        
        bboxes_by_frame.append(frame_bboxes)
        keypoints_by_frame.append(frame_keypoints)
        prev_bboxes = frame_bboxes
        # print(frame_bboxes)
        # print(frame_keypoints)
        
        # if frame_idx > 15:        
        #     break
    
    vid_keypoints_tmp = np.array(keypoints_by_frame)
    vid_bboxes_tmp = np.array(bboxes_by_frame)
    with_confidence = 1
    if with_confidence == 0:
        vid_keypoints_tmp = vid_keypoints_tmp[:, :, :, :2]
        vid_bboxes_tmp = vid_bboxes_tmp[:, :, :4]
    
    # print(vid_keypoints_tmp.shape)
    # print(vid_bboxes_tmp.shape)
    vid_keypoints_tmp = np.reshape(vid_keypoints_tmp, (-1, 2*21*(2+with_confidence)))
    vid_bboxes_tmp = np.reshape(vid_bboxes_tmp, (-1, 2*(4+with_confidence)))
    vid_poses_tmp = np.concatenate((vid_keypoints_tmp, vid_bboxes_tmp), axis=1)
    # print(vid_poses_tmp.shape)
    return vid_poses_tmp.transpose()


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


def pose_accelleration(keypoints1:np.ndarray, keypoints2:np.ndarray, keypoints3:np.ndarray):
    """_summary_

    Args:
        keypoints1 (np.ndarray): of shape 21, 3
        keypoints2 (np.ndarray): of shape 21, 3

    Returns:
        np.ndarray: of shape 21,2
    """
    x_v1 = velocity(keypoints1[:, 0], keypoints2[:, 0])
    x_v2 = velocity(keypoints2[:, 0], keypoints3[:, 0])
    x = velocity(x_v1, x_v2)
    x = x.reshape(21, 1)
    # print(x.shape)
    
    y_v1 = velocity(keypoints1[:, 1], keypoints2[:, 1])
    y_v2 = velocity(keypoints2[:, 1], keypoints3[:, 1])
    y = velocity(y_v1, y_v2)
    y = y.reshape(21, 1)
    # print(y.shape)
    return np.concatenate((x, y), axis=1)

def combine_poses_with_detections(vid_poses, vid_detections):
    # print(vid_poses.shape)
    # print(vid_detections.shape)
    
    vid_bboxes_without_hands = vid_detections[2*5:, :]
    combined = np.concatenate((vid_poses, vid_bboxes_without_hands), axis=0)
    return combined
    


def convert_combined_poses_and_detections_to_centers(combined_features):
    
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
    left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools = load_raw_combined(combined_features)
    converted = []
    for frame_id in range(combined_features.shape[1]):
        _left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools = get_combined_for_frame(left_keypoints, left_bbox, right_keypoints, right_bbox, vid_tools, frame_id)
        converted.append(convert_frame_features(_left_keypoints, _left_bbox, _right_keypoints, _right_bbox, _vid_tools))
    converted = np.array(converted)
    return converted.transpose()















#################### jupyter notebook utils ################


from scipy import signal

def savgol_filter(keypoints):
    keypoints = keypoints.copy()
    window_length, polyorder = 13, 2
    
    for i in range(keypoints.shape[1]): 
        keypoints[:, i, 0] = signal.savgol_filter(keypoints[:, i, 0], window_length, polyorder)
        keypoints[:, i, 1] = signal.savgol_filter(keypoints[:, i, 1], window_length, polyorder)
    return keypoints

