from typing import List

from src.models.composite.late_fusion_tcn2 import LF_MH_MS_TCN2
from src.models.external.multii_head_tcn2 import MH_MS_TCN2
from src.models.external.tcn2 import MS_TCN2

def get_multi_task_tcn2_model(features_dim: int, num_classeses: List[int], num_f_maps:int, 
                              num_layers_PG:int, num_layers_R:int, num_R: int):
    return MH_MS_TCN2(num_layers_PG=num_layers_PG,
                             num_layers_R=num_layers_R,
                             num_R=num_R,
                             num_f_maps=num_f_maps,
                             dim=features_dim,
                             num_classeses=num_classeses)

def get_tcn2_model(features_dim: int, num_classes: int, num_f_maps:int=64, 
                              num_layers_PG:int=11, num_layers_R:int=10, num_R: int=3, dropout=0.5):
    return MS_TCN2(num_layers_PG=num_layers_PG,
                             num_layers_R=num_layers_R,
                             num_R=num_R,
                             num_f_maps=num_f_maps,
                             dim=features_dim,
                             num_classes=num_classes,
                             dropout=dropout)

def get_late_fused_mh_tcn2_model(features_dim: int, num_classeses: List[int], num_f_maps:int, 
                              num_layers_PG:int, num_layers_R:int, num_R: int):
    num_gestures = num_classeses[0]
    assert(num_gestures == 6)
    num_tools_left = num_classeses[1]
    left = get_tcn2_model(features_dim, num_tools_left, num_f_maps, 
                              num_layers_PG, num_layers_R, num_R)
    num_tools_right = num_classeses[2]
    right = get_tcn2_model(features_dim, num_tools_right, num_f_maps, 
                              num_layers_PG, num_layers_R, num_R)
    fused = get_tcn2_model(features_dim + num_tools_left + num_tools_right, num_gestures, num_f_maps, 
                              num_layers_PG, num_layers_R, num_R)
    return LF_MH_MS_TCN2(fused, left, right)
