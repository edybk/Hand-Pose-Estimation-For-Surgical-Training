
from dataclasses import dataclass
import time
from typing import List, Union, Optional
import torch
from torch import nn

@dataclass
class ModelOutputConfig:
    model: nn.Module
    input_data: str
    sample_rate: int
    

@dataclass
class ModelConfig:
    model: str
    features_dim: int
    num_classes: int
    num_f_maps: int
    num_layers_PG: int
    num_layers_R: int
    num_R: int
    actions_dict: dict
    actions_dict2: dict
    sample_rate: int 
    dataset: str
    dropout:float = 0.5
    ckpt: str = None

@dataclass
class TrainConfig:
    device: torch.device
    model_dir: str
    num_epochs: int
    gt_path: str
    split: str
    model:nn.Module
    actions_dict:dict
    dataset:str
    features_path:str
    log_dir:str
    gt_path_tools_left:str
    gt_path_tools_right:str
    actions_dict2:dict
    model_dir:str
    num_classes:Union[int, List[int]]
    resume:str = None
    bz:int = 1
    lr: float = 0.0005
    input_data: str = "raw"
    eval_rate: int = 10
    smoothing_loss_param: float = 0.15
    class_weights: Optional[List[float]] = None
    weight_decay: Optional[float] = None
    benchmark_durations: Optional[bool] = False
    

  
@dataclass
class BatchGeneratorConfig:
    num_classes:int
    actions_dict:dict
    gt_path:str
    features_path:str
    sample_rate:int
    gt_path_tools_left:str = None
    gt_path_tools_right:str = None
    actions_dict2:dict = None
    appended_features: Optional[list] = None
    excluded_participants: Optional[list] = None
    split:Optional[Union[str, int]] = None
