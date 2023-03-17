import numpy as np
from src.models.pose.base import PoseResult, TwoHandPoseResult

from src.models.pose_enhancer.base import PoseEnhancer, TwoHandPoseEnhancer

class RawPoseEnhancer(PoseEnhancer):
    
    def forward(self, x:PoseResult):
        return np.concatenate((x.keypoints.flatten(), x.bboxes.flatten()))
    
class TwoHandRawPoseEnhancer(TwoHandPoseEnhancer):
    
    def forward(self, x:TwoHandPoseResult):
        return np.concatenate((x.left.keypoints.flatten(), 
                              x.left.bboxes.flatten(), 
                              x.right.keypoints.flatten(), 
                              x.right.bboxes.flatten()))