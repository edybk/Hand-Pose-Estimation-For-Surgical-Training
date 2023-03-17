
import os
from src.models.detection.mmdet_base import MMDetModel
from src.models.pose.mmpose_base import MMPoseModel, TwoHandsMMPoseModel



class ResnetPoseModel(MMPoseModel):
    """[summary]
    Uses mmdet to detect bounding boxes and mmpose resnet pretrained model to detect poses.
    Poses are of dimension 21x3 where the third element is the confidence threshold.
    """
    
    def __init__(self, det_model:MMDetModel):
        repository_root = os.getenv("REPOSITORY_ROOT")
        pose_config = repository_root + "/pose/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/res50_apas_224x224.py"
        pose_checkpoint = repository_root + "/pose/mmpose/work_dirs/res50_apas_224x224/epoch_59.pth"
        super(ResnetPoseModel, self).__init__(det_model=det_model, pose_config=pose_config, pose_checkpoint=pose_checkpoint)


class TwoHandsResnetPoseModel(TwoHandsMMPoseModel):
    """[summary]
    Uses mmdet to detect bounding boxes and mmpose resnet pretrained model to detect poses.
    Poses are of dimension 21x3 where the third element is the confidence threshold.
    """
    
    def __init__(self, det_model:MMDetModel):
    
        repository_root = os.getenv("REPOSITORY_ROOT")
        pose_config = repository_root + "/pose/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/res50_apas_224x224.py"
        pose_checkpoint = repository_root + "/pose/mmpose/work_dirs/res50_apas_224x224/epoch_59.pth"
        super(TwoHandsResnetPoseModel, self).__init__(det_model=det_model, pose_config=pose_config, pose_checkpoint=pose_checkpoint)
