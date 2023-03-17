from abc import ABC, abstractmethod
from dataclasses import dataclass
    
class DetectionModel(ABC):
    
    @abstractmethod
    def get_det_model(self):
        pass
    
    def detect(self, img):
        pass
    
    def detect_one_per_category(self, img, thresh=0.5):
        pass