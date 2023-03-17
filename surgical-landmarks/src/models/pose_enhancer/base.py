from abc import ABC, abstractmethod

from numpy import ndarray

from src.models.pose.base import PoseResult, TwoHandPoseResult

class PoseEnhancer(ABC):
    """[summary]
    This model is responsible for making features out of pose estimations
    Input: PoseResult
    Output: ???
    """
    
    @abstractmethod
    def forward(self, pose_result:PoseResult) -> ndarray:
        pass
    
class TwoHandPoseEnhancer(ABC):
    """[summary]
    This model is responsible for making features out of pose estimations
    Input: PoseResult
    Output: ???
    """
    
    @abstractmethod
    def forward(self, pose_result:TwoHandPoseResult) -> ndarray:
        pass