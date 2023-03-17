from src.models.detection.mmdet_base import MMDetModel

import os

class AllClassAPASYOLOXDetModel(MMDetModel):
    """[summary]
    Uses mmdet to detect bounding boxes
    """
    
    def __init__(self, detection_categories=None):
        repository_root = os.getenv("REPOSITORY_ROOT")
        det_config = repository_root + "/detection/mmdetection/configs/yolox/yolox_s_8x8_300e_apas_allclass.py"
        det_checkpoint = repository_root + "/detection/mmdetection/work_dirs/yolox_s_8x8_300e_apas_allclass/epoch_690.pth"

        """
        {"mode": "val", "epoch": 690, "iter": 132, "lr": 0.00746, "bbox_mAP": 0.719, "bbox_mAP_50": 0.946, "bbox_mAP_75": 0.8, "bbox_mAP_s": 0.441, "bbox_mAP_m": 0.76, "bbox_mAP_l": 0.854, "bbox_mAP_copypaste": "0.719 0.946 0.800 0.441 0.760 0.854"}
        """
        super(AllClassAPASYOLOXDetModel, self).__init__(det_config=det_config, det_checkpoint=det_checkpoint, detection_categories=detection_categories)
