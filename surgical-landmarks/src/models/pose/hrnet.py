
import cv2
import numpy as np
import os
from src.models.detection.mmdet_base import MMDetModel
from src.models.pose.mmpose_base import MMPoseModel



class HRNetPoseModel(MMPoseModel):
    """[summary]
    Uses mmdet to detect bounding boxes and mmpose HRNet pretrained model to detect poses.
    Poses are of dimension 21x3 where the third element is the confidence threshold.
    """
    
    def __init__(self, det_model:MMDetModel):
        repository_root = os.getenv("REPOSITORY_ROOT")
        pose_config = repository_root + "/pose/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/hrnetv2_w18_apas_256x256.py"
        pose_checkpoint = repository_root + "/pose/mmpose/work_dirs/hrnetv2_w18_apas_256x256/epoch_113.pth"
        super(HRNetPoseModel, self).__init__(det_model=det_model, pose_config=pose_config, pose_checkpoint=pose_checkpoint)