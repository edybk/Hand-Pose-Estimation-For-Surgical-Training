from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from numpy import ndarray

@dataclass
class PoseResult:
    # N is number of hands, last value is confidence
    keypoints: ndarray #[N, 21, 3]
    bboxes: ndarray # [N, 5]
    img: Optional[any] = None
    backbone: Optional[ndarray] = None
    
@dataclass
class HandPoseResult:
    keypoints: ndarray #[21, 3]
    bboxes: ndarray # [5]
    
@dataclass
class TwoHandPoseResult:
    left: HandPoseResult
    right: HandPoseResult
    img: Optional[any] = None
    backbone: Optional[ndarray] = None
    
class PoseModel(ABC):
    """[summary]
    This model is responsible for taking RGB images as input and predicting hand poses and bbox as output.
    Input: [3, W, H]
    Output: PoseResult where N is the number of hands
    """
    
    @abstractmethod
    def forward(self, x: ndarray, visualize:bool=False) -> PoseResult:
        pass
    
class TwoHandPoseModel(ABC):
    """[summary]
    This model is responsible for taking RGB images as input and predicting hand poses and bbox as output.
    Input: [3, W, H]
    Output: PoseResult where N is the number of hands
    """
    
    @abstractmethod
    def forward(self, x: ndarray, visualize:bool=False) -> TwoHandPoseResult:
        pass