from abc import ABC, abstractmethod

from numpy import ndarray


class DetectionEnhancer(ABC):
    """[summary]
    This model is responsible for making features out of detections
    Input: detections
    Output: ???
    """
    
    @abstractmethod
    def forward(self, det_result:dict) -> ndarray:
        pass
    