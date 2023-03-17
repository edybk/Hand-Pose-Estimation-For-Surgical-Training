from src.models.detection.mmdet_base import MMDetModel
import os

class TwoHandAPASYOLOXDetModel(MMDetModel):
    """[summary]
    Uses mmdet to detect bounding boxes
    """
    
    def __init__(self):
        repository_root = os.getenv("REPOSITORY_ROOT")
        det_config = repository_root + "/detection/mmdetection/configs/yolox/yolox_s_8x8_300e_apas_2hands_max2hands.py"
        det_checkpoint = repository_root + "/detection/mmdetection/work_dirs/yolox_s_8x8_300e_apas_2hands_max2hands/best_bbox_mAP_epoch_970.pth"
        """
        {"mode": "val", "epoch": 970, "iter": 131, "lr": 0.00544, "bbox_mAP": 0.905, "bbox_mAP_50": 1.0, "bbox_mAP_75": 0.994, "bbox_mAP_s": -1.0, "bbox_mAP_m": 0.893, "bbox_mAP_l": 0.923, "bbox_mAP_copypaste": "0.905 1.000 0.994 -1.000 0.893 0.923"}
        """
        super(TwoHandAPASYOLOXDetModel, self).__init__(det_config=det_config, det_checkpoint=det_checkpoint)
