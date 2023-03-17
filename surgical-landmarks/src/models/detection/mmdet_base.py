
from src.models.detection.base import DetectionModel
import numpy as np
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
class MMDetModel(DetectionModel):
    """[summary]
    Uses mmdet to detect bounding boxes
    """
    
    def __init__(self, det_config, det_checkpoint, detection_categories = None):
        self.device = "cuda:0"
        assert has_mmdet, 'Please install mmdet to run the demo.'

        assert det_config is not None
        assert det_checkpoint is not None

        self.det_model = init_detector(
            det_config, det_checkpoint, device=self.device.lower())
        self.detection_categories = detection_categories

    def get_det_model(self):
        return self.det_model

    def detect(self, img):
        mmdet_results = inference_detector(self.det_model, img)
        return mmdet_results
    
    def detect_one_per_category(self, img, visualize=False, thresh=0.5):
        mmdet_results = inference_detector(self.det_model, img)
        # filtered_results = []
        results_per_category = {}
        for i, detections in enumerate(mmdet_results):
            if self.detection_categories is not None and i not in self.detection_categories:
                continue
            
            # print(detections.shape)
            if detections.shape[0] > 0:
                max_detection = detections[np.argmax(detections[:, 4])]
                max_detection = max_detection if max_detection[4] > thresh else np.zeros(5)
            else:
                max_detection = np.zeros(5)
            
            results_per_category[i+1] = max_detection
            # filtered_results.append(np.expand_dims(max_detection, axis=0))
        frame = None
        if visualize:
            frame = self.det_model.show_result(img, mmdet_results, score_thr=thresh)
        return results_per_category, frame