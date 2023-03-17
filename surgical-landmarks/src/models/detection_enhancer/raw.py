import numpy as np
from src.models.detection_enhancer.base import DetectionEnhancer

class RawDetectionEnhancer(DetectionEnhancer):
    def forward(self, x:dict):
        return np.concatenate([x[k].flatten() for k in sorted(list(x.keys()))])
    