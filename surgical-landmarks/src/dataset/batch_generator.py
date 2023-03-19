import torch
import numpy as np
import random
from src.configs.base import BatchGeneratorConfig

class BatchGenerator(object):
    
    def __init__(self, cfg:BatchGeneratorConfig):
        
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = cfg.num_classes[0] if isinstance(cfg.num_classes, list) else cfg.num_classes
        self.actions_dict = cfg.actions_dict
        self.gt_path = cfg.gt_path
        self.features_path = cfg.features_path
        self.sample_rate = cfg.sample_rate        
        
        self.gt_path_tools_left = cfg.gt_path_tools_left
        self.gt_path_tools_right = cfg.gt_path_tools_right
        self.actions_dict2 = cfg.actions_dict2
        self.num_classes2 = len(cfg.actions_dict2) if cfg.actions_dict2 else 0
        # print(actions_dict2)
                
        self.appended_features = cfg.appended_features
                
        self.excluded_participants = cfg.excluded_participants
        
        self.split = cfg.split
        
      
    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_sample(self, sample_name:str):
        """[summary]

        Args:
            vid_list_file (str): path to file containing names of video files including extension, one on each line
            e.g.: 
                vid1.mp4
                vid2.mp4
                ...
        """
        self.list_of_examples = [sample_name+".txt"]

    def filter_participants_if_needed(self, l):
        if not self.excluded_participants:
            return l
        for ex in self.excluded_participants:
            l = [x for x in l if ex not in x]
        return l

    def read_data(self, vid_list_file:str):
        """[summary]

        Args:
            vid_list_file (str): path to file containing names of video files including extension, one on each line
            e.g.: 
                vid1.mp4
                vid2.mp4
                ...
        """
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        
        self.list_of_examples = self.filter_participants_if_needed(self.list_of_examples)
        random.shuffle(self.list_of_examples)


    def load_features_from_path(self, features_path, vid_name):
        fp = features_path
        if features_path.endswith("_split"):
            fp = fp + str(self.split)
        return np.load(fp + vid_name + '.npy') #get features for video from features_path/vid_name{withoutextension}.npy

    def append_features(self, features, appended_features, vid_name):
        final_features = [features]
        final_features_lens = [features.shape[1]]
        # print(features.shape)
        for afeatures in appended_features:
            afeatures_array = self.load_features_from_path(afeatures, vid_name)
            # print(afeatures_array.shape)
            final_features.append(afeatures_array)
            final_features_lens.append(afeatures_array.shape[1])
        length = min(final_features_lens)
        final_features = [f[:,:length] for f in final_features]
        # print([f.shape for f in final_features])
        return np.concatenate(final_features, axis=0)
        
        
    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []

            
        if self.actions_dict2:
            batch_target_tools_left = []
            batch_target_tools_right = []
            
        for vid in batch:
            vid_name = vid.split('.')[0]
            features = self.load_features_from_path(self.features_path,vid_name)
            
            if self.appended_features:
                features = self.append_features(features, self.appended_features, vid_name)
            
            with open(self.gt_path + vid, 'r') as file_ptr:
                content = file_ptr.read().split('\n')[:-1]
            
            
            length = min(np.shape(features)[1], len(content))
            # print(length)
            if self.actions_dict2:
                with open(self.gt_path_tools_left + vid, 'r') as file_ptr:
                    content_tools_left = file_ptr.read().split('\n')[:-1]
                with open(self.gt_path_tools_right + vid, 'r') as file_ptr:
                    content_tools_right = file_ptr.read().split('\n')[:-1]
                length = min(length , len(content_tools_left), len(content_tools_right))
                classes_tools_left = np.zeros(length)
                classes_tools_right = np.zeros(length)
            
            features = features[:,:length]
            classes = np.zeros(length)
            
            #new
            
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            

            
            sampled_features = features[:, ::self.sample_rate]
            # print(f"sampled_features {sampled_features.shape}")
            
            sampled_targets = classes[::self.sample_rate]
                
            # print(f"sampled_targets {sampled_targets.shape}")
    
            batch_input.append(sampled_features)
            batch_target.append(sampled_targets)

            
            if self.actions_dict2:


                for i in range(len(classes)):
                    # print(f"len classes_tools_left {len(classes_tools_left)} ")
                    # print(f"len content_tools_left {len(content_tools_left)} ")
                    classes_tools_left[i] = self.actions_dict2[content_tools_left[i]]
                    classes_tools_right[i] = self.actions_dict2[content_tools_right[i]]
                

                sampled_targets_tools_left = classes_tools_left[::self.sample_rate]
                sampled_targets_tools_right = classes_tools_right[::self.sample_rate]
                    
                batch_target_tools_left.append(sampled_targets_tools_left)
                batch_target_tools_right.append(sampled_targets_tools_right)
                
            

        length_of_sequences = list(map(len, batch_target))
        max_sequence_length = max(length_of_sequences)

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_sequence_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_sequence_length, dtype=torch.long)*(-100)
        
        if self.actions_dict2:
            batch_target_tensor_tools_left = torch.ones(len(batch_input), max_sequence_length, dtype=torch.long)*(-100)
            batch_target_tensor_tools_right = torch.ones(len(batch_input), max_sequence_length, dtype=torch.long)*(-100)
            mask_tl = torch.zeros(len(batch_input), self.num_classes2, max_sequence_length, dtype=torch.float)
            mask_tr = torch.zeros(len(batch_input), self.num_classes2, max_sequence_length, dtype=torch.float)
        
        
        mask = torch.zeros(len(batch_input), self.num_classes, max_sequence_length, dtype=torch.float)
        
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            
            if self.actions_dict2:
                batch_target_tensor_tools_left[i, :np.shape(batch_target_tools_left[i])[0]] = torch.from_numpy(batch_target_tools_left[i])
                batch_target_tensor_tools_right[i, :np.shape(batch_target_tools_right[i])[0]] = torch.from_numpy(batch_target_tools_right[i])
                mask_tl[i, :, :np.shape(batch_target_tools_left[i])[0]] = torch.ones(self.num_classes2, np.shape(batch_target_tools_left[i])[0])
                mask_tr[i, :, :np.shape(batch_target_tools_right[i])[0]] = torch.ones(self.num_classes2, np.shape(batch_target_tools_right[i])[0])
            
            

        # print(batch_input_tensor.shape)
        
        res =  {
            "input": batch_input_tensor, 
            "target": batch_target_tensor, 
            "mask": mask, 
            "vids": batch
        }


        if self.actions_dict2:
            res["target_tools_left"] = batch_target_tensor_tools_left
            res["target_tools_right"] = batch_target_tensor_tools_right
            res["mask_tl"] = mask_tl
            res["mask_tr"] = mask_tr
        
        return res