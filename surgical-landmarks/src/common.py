import json
from typing import Optional
from src.configs.base import ModelConfig
from src.models.model import get_late_fused_mh_tcn2_model, get_multi_task_tcn2_model, get_tcn2_model
import random
import torch
from torch import nn

def load_json(file_path:str):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ckpt(model: nn.Module, ckpt:str):
    model.load_state_dict(torch.load(ckpt))

def init_gt(dataset:str, multitask:bool, task_left_only:bool, task_right_only:bool, task_custom: Optional[str] = None, given_mapping_file=None):
    gt_path = "./data/"+dataset+"/groundTruth/"
    mapping_file = "./data/"+dataset+"/mapping.txt"
    gt_path_tools_left = None
    gt_path_tools_right = None
    mapping_file2 = None
    actions_dict2 = None
    if multitask:
        gt_path_tools_left = "./data/"+dataset+"/groundTruthToolsLeft/"
        gt_path_tools_right = "./data/"+dataset+"/groundTruthToolsRight/"
        mapping_file2 = "./data/"+dataset+"/mapping_tools.txt"
        actions_dict2 = init_actions_dict(mapping_file2)
    elif task_left_only:
        mapping_file = "./data/"+dataset+"/mapping_tools.txt"
        gt_path = "./data/"+dataset+"/groundTruthToolsLeft/"
    elif task_right_only:
        mapping_file = "./data/"+dataset+"/mapping_tools.txt"
        gt_path = "./data/"+dataset+"/groundTruthToolsRight/"
    elif task_custom:
        mapping_file = "./data/"+dataset+"/mapping.txt"
        gt_path = "./data/"+dataset+"/" + task_custom + "/"

    if given_mapping_file:
        mapping_file = "./data/"+dataset+'/' + given_mapping_file
        
    actions_dict = init_actions_dict(mapping_file)
    if actions_dict2:
        num_classes = [len(actions_dict), len(actions_dict2), len(actions_dict2)]
    else:    
        num_classes = len(actions_dict)
    
    return gt_path, actions_dict, gt_path_tools_left, gt_path_tools_right, mapping_file2, actions_dict2, num_classes

def init_vid_list_files(dataset:str, split:str):
    
    vid_list_file_train = "./data/"+dataset+"/splits/train.split"+split+".bundle"
    vid_list_file_val = "./data/"+dataset+"/splits/val.split"+split+".bundle"
    vid_list_file_tst = "./data/"+dataset+"/splits/test.split"+split+".bundle"
    return {
        "train": vid_list_file_train,
        "val": vid_list_file_val,
        "test": vid_list_file_tst
    }

def init_actions_dict(mapping_file:str):
    with open(mapping_file, 'r') as file_ptr:
        actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return actions_dict

def init_features_path(dataset:str, flags:dict):
    features_path = "./data/"+dataset+"/features/"
    if flags.get("custom_features"):
        custom = flags["custom_features"]
        features_path = "./data/"+dataset+f"/{custom}/"
    return features_path

def init_model(cfg:ModelConfig, train_cfg_dict:dict, batch_gen_cfg_dict:dict):
    model = None
    if cfg.model == "TCN2":
        model=get_tcn2_model(cfg.features_dim, cfg.num_classes,
                         num_f_maps=cfg.num_f_maps,
                         num_layers_PG=cfg.num_layers_PG,
                         num_layers_R=cfg.num_layers_R,
                         num_R=cfg.num_R,
                         dropout=cfg.dropout)
    elif cfg.model == "MHTCN2":
        model = get_multi_task_tcn2_model(cfg.features_dim, 
                                        num_classeses=[
                                            len(cfg.actions_dict), 
                                            len(cfg.actions_dict2), 
                                            len(cfg.actions_dict2)],
                                        num_f_maps=cfg.num_f_maps,
                                        num_layers_PG=cfg.num_layers_PG,
                                        num_layers_R=cfg.num_layers_R,
                                        num_R=cfg.num_R)
    elif cfg.model == "LFMHTCN2":
        model = get_late_fused_mh_tcn2_model(cfg.features_dim, 
                                        num_classeses=[
                                            len(cfg.actions_dict), 
                                            len(cfg.actions_dict2), 
                                            len(cfg.actions_dict2)],
                                        num_f_maps=cfg.num_f_maps,
                                        num_layers_PG=cfg.num_layers_PG,
                                        num_layers_R=cfg.num_layers_R,
                                        num_R=cfg.num_R)
    else:
        raise "UNKNOWN MODEL"
    
    
    if cfg.ckpt:
        load_ckpt(model, cfg.ckpt)
    train_cfg_dict["model"] = model
    return model

    
def init_seed(seed = 1538574472):
    ### Seed Initialization
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ###