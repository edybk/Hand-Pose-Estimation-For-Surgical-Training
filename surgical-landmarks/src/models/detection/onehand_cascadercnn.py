from src.models.detection.mmdet_base import MMDetModel
import os

class OnehandCascadeRCNNDetModel(MMDetModel):
    """[summary]
    Uses mmdet to detect bounding boxes
    """
    
    def __init__(self):
        repository_root = os.getenv("REPOSITORY_ROOT")
        det_config = repository_root + "/pose/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py"
        det_checkpoint = "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"
        super(OnehandCascadeRCNNDetModel, self).__init__(det_config=det_config, det_checkpoint=det_checkpoint)
