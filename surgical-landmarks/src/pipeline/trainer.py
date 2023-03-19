from collections import defaultdict
import sys
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import os
from loguru import logger
import wandb
from src.common import load_ckpt
from src.configs.base import TrainConfig
from src.dataset.batch_generator import BatchGenerator
import time

from src.pipeline.metrics import *
from src.pipeline.util import Timer

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


class Trainer:
    
    def cross_entropy_loss(self, ignore_index, class_weights=None, device=None):
        weights = torch.Tensor(class_weights).to(device) if class_weights else None
        return nn.CrossEntropyLoss(weights, ignore_index=ignore_index)

    DEFAULT_CFG = {
        'smoothing_loss_param': 0.15
        # 'smoothing_loss_param': 0.05
    }
    
    def __init__(self, cfg: TrainConfig):
        self.model = cfg.model 
        
        self.ce = self.cross_entropy_loss(-100, cfg.class_weights, cfg.device)
        
        self.mse = nn.MSELoss(reduction='none')
        self.actions_dict = cfg.actions_dict
        self.num_classes = cfg.num_classes
        self.features_path = cfg.features_path
        self.ground_truth_path = cfg.gt_path
        log_name = 'logs/' + cfg.dataset + "_" + cfg.split + f"_{time.time()}.log"
        logger.remove()
        logger.add(log_name)
        logger.add(sys.stdout, colorize=True, format="{message}")
        logger.info(f"log location {log_name}")
        self.actions_dict2 = cfg.actions_dict2
        
        if self.actions_dict2:
            self.num_classeses = cfg.num_classes
        self.ground_truth_path_tools_left = cfg.gt_path_tools_left
        self.ground_truth_path_tools_right = cfg.gt_path_tools_right
        self.current_split = cfg.split
        self.cfg = cfg
        self.wandb_prefix = f"split{cfg.split}_"
 

    def _calc_loss(self, predictions, batch_target, loss, mask, batch_target_tools_left, batch_target_tools_right, mask_tl, mask_tr):
        if batch_target_tools_left is None or batch_target_tools_right is None:
            for p in predictions:
                # print(f"p shape {p.shape}")
                # print(p[:,:,0])
                ce_pred = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
                # print(np.histogram(ce_pred.argmax(1).cpu(), bins=range(7)))
                ce_target = batch_target.view(-1)
                # print(f"ce pred {ce_pred.shape} ce_target {ce_target.shape}")
                # print(f"ce_pred sum {ce_pred.sum()}, ")
                ce_loss = self.ce(ce_pred, ce_target)
                # print(f"ce_loss: {ce_loss}")
                
                loss += ce_loss
                smoothing_loss = self.smoothing_loss_param*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                # print(f"smoothing_loss: {smoothing_loss}")
                loss += smoothing_loss
        else:
            
            targetsss = [batch_target, batch_target_tools_left, batch_target_tools_right]
            masksss = [mask, mask_tl, mask_tr]
            # print(f"mask shape: {mask.shape}")
            for i in range(3):
                _predictions = predictions[i]
                num_classes = self.num_classeses[i]
                target = targetsss[i]
                _mask = masksss[i]
                for p in _predictions:
                    # print(p.shape)
                    ce_pred = p.transpose(2, 1).contiguous().view(-1, num_classes)
                    ce_target = target.view(-1)
                    ce_loss = self.ce(ce_pred, ce_target)
                    loss += ce_loss
                    smoothing_loss = self.smoothing_loss_param*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*_mask[:, :, 1:])
                    loss += smoothing_loss
        # print(f"loss: {loss}")
        
        return loss
    
    def _forward(self, batch_input, mask, device):
        batch_input = batch_input.to(device)
        predictions = self.model(batch_input, mask)
        return predictions
    
    def _calc_accuracy(self, predictions, batch_target, mask, correct, total,
                       batch_target_tools_left, batch_target_tools_right,
                       correct_tl, correct_tr, mask_tl, mask_tr):
        
        def _calc_accuracy_aux(ppredictions, target, mask):
            _, predicted = torch.max(ppredictions[-1].data, 1)
            correct = ((predicted == target).float()*mask[:, 0, :].squeeze(1)).sum().item()
            total = torch.sum(mask[:, 0, :]).item()
            return correct, total
        
        if batch_target_tools_left is None or batch_target_tools_right is None:
            c, t = _calc_accuracy_aux(predictions, batch_target, mask)
            correct += c
            total += t
            return correct, total, 0, 0
        else:
            c, t = _calc_accuracy_aux(predictions[0], batch_target, mask)
            correct += c
            total += t
            
            ctl, _ = _calc_accuracy_aux(predictions[1], batch_target_tools_left, mask_tl)
            ctr, _ = _calc_accuracy_aux(predictions[2], batch_target_tools_right, mask_tr)
            correct_tl += ctl
            correct_tr += ctr
            return correct, total, correct_tl, correct_tr

    def load_ckpt(self, ckpt):
        load_ckpt(self.model, ckpt)
        

    def train(self, cfg:TrainConfig, batch_gen:BatchGenerator, val_batch_gen: BatchGenerator, test_batch_gen: Optional[BatchGenerator] = None):
        if cfg.resume:
            self.model.load_state_dict(torch.load(cfg.resume))
        self.model.to(cfg.device)

        self.smoothing_loss_param = cfg.smoothing_loss_param
        optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        if cfg.weight_decay:
            optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        epoch = 0

        if cfg.resume:
            nm = '.'.join(cfg.resume.split('.')[:-1])
            optimizer.load_state_dict(torch.load(nm+'.opt'))
            epoch = int(nm.split('-')[-1])
        
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        self.model.train()
        print(f"Starting train")
        print(f"Train samples: {len(batch_gen.list_of_examples)}")
        print(f"Val samples: {len(val_batch_gen.list_of_examples)}")
        print(f"Model: {self.model}")
        
        best_mean = 0
        best_std = 0
        best_epoch = 0
        last_model_save_location = None
        last_opt_save_location = None
        
        while epoch < cfg.num_epochs:
            wandb.log({
                self.wandb_prefix + 'epoch': epoch+1
            })
            epoch_loss = 0
            correct = 0
            total = 0
            
            correct_tl = 0
            correct_tr = 0
            
            with Timer("epoch_train_time", verbose = cfg.benchmark_durations) as _:
                while batch_gen.has_next():
                    with Timer("next_batch") as _:
                        bt = batch_gen.next_batch(cfg.bz)
                        batch_input = bt.get("input")
                        batch_target = bt.get("target")
                        mask = bt.get("mask")
                        batch_target_tools_left = bt.get("target_tools_left")
                        batch_target_tools_right = bt.get("target_tools_right")
                        mask_tl = bt.get("mask_tl")
                        mask_tr = bt.get("mask_tr")
                        # batch_input, batch_target, mask, vids, \
                        #     frames, second_batch_input, batch_target_tools_left, batch_target_tools_right, mask_tl, mask_tr = \
                        #     bt.get("input"), bt.get("target"), bt.get("mask"), \
                        #         bt.get("vids"), bt.get("frames"), bt.get("second_input"), \
                        #             bt.get("target_tools_left"), bt.get("target_tools_right"), \
                        #                 bt.get("mask_tl"), bt.get("mask_tr")
                    # batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                    batch_target, mask = batch_target.to(cfg.device), mask.to(cfg.device)
                    optimizer.zero_grad()
                    
                    with Timer("forward") as _:
                        predictions = self._forward(batch_input, mask, cfg.device)
                    
                    if batch_target_tools_left is not None and batch_target_tools_right is not None and \
                        mask_tl is not None and mask_tr is not None:
                        batch_target_tools_left = batch_target_tools_left.to(cfg.device)
                        batch_target_tools_right = batch_target_tools_right.to(cfg.device)
                        mask_tl, mask_tr = mask_tl.to(cfg.device), mask_tr.to(cfg.device)
                    
                    loss = 0
                    with Timer("loss") as _:
                        loss = self._calc_loss(predictions, batch_target, loss, mask, batch_target_tools_left, batch_target_tools_right, mask_tl, mask_tr)
                        
                    epoch_loss += loss.item()
                    
                    with Timer("backward") as _:
                        loss.backward()
                    optimizer.step()

                    with Timer("accuracy") as _:
                        correct, total, correct_tl, correct_tr = self._calc_accuracy(predictions, batch_target, mask, correct, total, batch_target_tools_left, batch_target_tools_right, correct_tl, correct_tr, mask_tl, mask_tr)

            scheduler.step(epoch_loss)
            batch_gen.reset()
            
            wandb.log({self.wandb_prefix + 'train_accuracy': 100*float(correct)/total, self.wandb_prefix + 'train_loss': epoch_loss / len(batch_gen.list_of_examples)})
            if not self.actions_dict2:
                logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))
            else:
                logger.info("[epoch %d]: epoch loss = %f,   acc_gesture = %f,   acc_tools_left = %f,    acc_tools_right = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total, float(correct_tl)/total, float(correct_tr)/total))
                wandb.log({self.wandb_prefix + 'train_accuracy_tools_left': 100*float(correct_tl)/total, self.wandb_prefix + 'train_accuracy_tools_right': 100*float(correct_tr)/total})
            if (epoch+1) % cfg.eval_rate == 0:
                
                logger.info("epoch: " + str(epoch+1) + " model evaluation")
                with Timer("eval") as _:
                    val_results_g, val_results_tl, val_results_tr = self.predict2(val_batch_gen, cfg.device, batch_gen.sample_rate, cfg.input_data, benchmark_durations=cfg.benchmark_durations)
                    acc, edit, cross_vid_accuracy_mean, cross_vid_accuracy_std = self.evaluate2(val_results_g, self.ground_truth_path)
                    wandb.log({self.wandb_prefix + 'validation_accuracy': acc, self.wandb_prefix + 'validation_edit': edit, self.wandb_prefix + 'validation_cross_vid_accuracy_mean': cross_vid_accuracy_mean, self.wandb_prefix + 'validation_cross_vid_accuracy_std': cross_vid_accuracy_std})

                    if cross_vid_accuracy_mean > best_mean:
                        best_mean = cross_vid_accuracy_mean
                        best_std = cross_vid_accuracy_std
                        logger.info(f"new best mean {best_mean} std {best_std} epoch {epoch+1}")
                        best_epoch = epoch + 1
                        with Timer("save ckpt") as _:
                            model_save_location = cfg.model_dir + "/epoch-" + str(epoch + 1) + ".model"
                            optimizer_save_location = cfg.model_dir + "/epoch-" + str(epoch + 1) + ".opt"
                            logger.info(f"saved model state to {model_save_location}")
                            torch.save(self.model.state_dict(), model_save_location)
                            torch.save(optimizer.state_dict(), optimizer_save_location)
                            if last_model_save_location:
                                # delete
                                try:
                                    os.remove(last_model_save_location)
                                    os.remove(last_opt_save_location)
                                except:
                                    pass
                            last_model_save_location = model_save_location
                            last_opt_save_location = optimizer_save_location
                            
                    if self.actions_dict2:
                        acc_tools_left, edit_tools_left, cross_vid_accuracy_mean_tools_left, cross_vid_accuracy_std_tools_left = self.evaluate2(val_results_tl, self.ground_truth_path_tools_left)
                        wandb.log({self.wandb_prefix + 'validation_accuracy_tools_left': acc_tools_left, self.wandb_prefix + 'validation_edit_tools_left': edit_tools_left, self.wandb_prefix + 'validation_cross_vid_accuracy_mean_tools_left': cross_vid_accuracy_mean_tools_left, self.wandb_prefix + 'validation_cross_vid_accuracy_std_tools_left': cross_vid_accuracy_std_tools_left})
                        acc_tools_right, edit_tools_right, cross_vid_accuracy_mean_tools_right, cross_vid_accuracy_std_tools_right = self.evaluate2(val_results_tr, self.ground_truth_path_tools_right)
                        wandb.log({self.wandb_prefix + 'validation_accuracy_tools_right': acc_tools_right, self.wandb_prefix + 'validation_edit_tools_right': edit_tools_right, self.wandb_prefix + 'validation_cross_vid_accuracy_mean_tools_right': cross_vid_accuracy_mean_tools_right, self.wandb_prefix + 'validation_cross_vid_accuracy_std_tools_right': cross_vid_accuracy_std_tools_right})
  
                self.model.train()
            epoch += 1
        print(f"Done training, split={self.current_split}, best cross vid accuracy mean={best_mean}, std={best_std}, epoch={best_epoch}")
        
        # TEST
        
        if test_batch_gen:
            logger.info(f"Testing with weights from epoch {best_epoch} - {last_model_save_location}")
            self.model.load_state_dict(torch.load(last_model_save_location))
            results_g, results_tl, results_tr = self.predict2(test_batch_gen, device=self.cfg.device, sample_rate=test_batch_gen.sample_rate, input_data=self.cfg.input_data, benchmark_durations=cfg.benchmark_durations)
            self.evaluate2(results_g, test_batch_gen.gt_path, verbose=True, wandb_prefix=f"Fixed_Test_split{self.current_split}")  
            
            if self.actions_dict2:
                self.evaluate2(results_tl, test_batch_gen.gt_path_tools_left, verbose=True, wandb_prefix=f"Fixed_Test_split{self.current_split}_tools_left")
                self.evaluate2(results_tr, test_batch_gen.gt_path_tools_right, verbose=True, wandb_prefix=f"Fixed_Test_split{self.current_split}_tools_right")
              

    def predict2(self, batch_gen:BatchGenerator, device, sample_rate, input_data, benchmark_durations = False):
        self.model.eval()
        results_g = {}
        results_tl = {}
        results_tr = {}
        with torch.no_grad():
            self.model.to(device)
            while batch_gen.has_next():
                bt = batch_gen.next_batch(1)
                batch_input = bt.get("input")
                batch_target = bt.get("target")
                mask = bt.get("mask")
                vids = bt.get("vids")
                # batch_input, batch_target, mask, vids, frames, second_batch_input, batch_target_tools_left, batch_target_tools_right,  mask_tl, mask_tr = \
                #     bt.get("input"), bt.get("target"), bt.get("mask"), bt.get("vids"), bt.get("frames"),  bt.get("second_input"),  bt.get("target_tools_left"), bt.get("target_tools_right"), bt.get("mask_tl"), bt.get("mask_tr")
                batch_target, mask = batch_target.to(device), mask.to(device)
                with Timer(f"inference_time_{int(batch_input.shape[-1])}", verbose = benchmark_durations):
                    predictions = self._forward(batch_input, mask, device)
                f_name = vids[0].split('/')[-1].split('.')[0]
                
                def _get_recognition_aux(ppredictions, actions_dict, filter_method=None):
                    if filter_method:
                        ppredictions = filter_method(ppredictions)
                    _, predicted = torch.max(ppredictions[-1].data, 1)
                    predicted = predicted.squeeze()
                    recognition = []
                    for i in range(len(predicted)):
                        recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                    return recognition
                
                if not self.actions_dict2:
                    
                    
                    filter_method = None
                    
                    f_results = _get_recognition_aux(predictions, self.actions_dict, filter_method=filter_method)
                    
                    results_g[f_name] = f_results
                    
                else:
                    results_g[f_name] = _get_recognition_aux(predictions[0], self.actions_dict)
                    results_tl[f_name] = _get_recognition_aux(predictions[1], self.actions_dict2)
                    results_tr[f_name] = _get_recognition_aux(predictions[2], self.actions_dict2)
                
            batch_gen.reset()
        return results_g, results_tl, results_tr


    def evaluate2(self, results, gt_path, verbose=False, wandb_prefix=None):
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
        correct = 0
        total = 0
        
        total_per_class = defaultdict(int)
        correct_per_class = defaultdict(int)
        per_vid_accuracy = defaultdict(float)
        per_vid_accuracy_per_class = {}
        edit = 0
        for vid in results.keys():
            gt_file = f"{gt_path}/{vid}.txt"
            gt_content = read_file(gt_file).split('\n')[0:-1]
            vid_name = vid.split('.')[0]
            recog_content = results[vid_name]
            
            vid_correct = 0
            vid_total = 0
            vid_total_per_class = defaultdict(int)
            vid_correct_per_class = defaultdict(int)
            
            for i in range(min(len(gt_content), len(recog_content))):
                gt_class = gt_content[i]
                vid_total_per_class[gt_class] += 1
                total_per_class[gt_class] += 1
                total += 1
                vid_total += 1
                gtci = gt_content[i]
                rcgci = recog_content[i]
                if gtci == rcgci:
                    correct += 1
                    vid_correct += 1
                    vid_correct_per_class[gt_class] += 1
                    correct_per_class[gt_class] += 1

            vid_accuracy = (100*float(vid_correct)/vid_total)
            per_vid_accuracy[vid] = vid_accuracy
            if verbose:
                logger.info(f"vid {vid} accuracy: {vid_accuracy}")
                vid_accuracy_per_class = {}
                for k in sorted(vid_total_per_class.keys()):
                    class_accuracy = (100*float(vid_correct_per_class[k])/vid_total_per_class[k])
                    # logger.info(f"class {k} accuracy: {class_accuracy}")
                    vid_accuracy_per_class[k] = class_accuracy
                per_vid_accuracy_per_class[vid] = vid_accuracy_per_class
                    
            
            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
            
        acc = (100*float(correct)/total)
        edit = ((1.0*edit)/len(results))
        logger.info("Acc: %.4f" % acc)
        logger.info('Edit: %.4f' % edit)
        if wandb_prefix:
            wandb.log({f"{wandb_prefix}_Acc": acc, f"{wandb_prefix}_Edit": edit})
        
        for k in sorted(total_per_class.keys()):
            class_accuracy = (100*float(correct_per_class[k])/total_per_class[k])
            logger.info(f"Total Class {k} Accuracy: {class_accuracy}")
            if wandb_prefix:
                wandb.log({f"{wandb_prefix}_Class_{k}_Acc": class_accuracy})
        
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            logger.info('F1@%0.2f: %.4f' % (overlap[s], f1))
            if wandb_prefix:
                wandb.log({f'{wandb_prefix}_F1@{overlap[s]}': f1})
            
            
        ### mean and std of accuracies per vid
        all_vid_accuracies = np.array(list(per_vid_accuracy.values()))
        cross_vid_accuracy_mean = np.mean(all_vid_accuracies)
        cross_vid_accuracy_std = np.std(all_vid_accuracies)
        logger.info("Cross Vid Accuracy Mean: %.4f" % cross_vid_accuracy_mean)
        logger.info("Cross Vid Accuracy STD : %.4f" % cross_vid_accuracy_std)
        
        if wandb_prefix:
            wandb.log({f'{wandb_prefix}_Cross_Vid_Accuracy_Mean': cross_vid_accuracy_mean,
                       f'{wandb_prefix}_Cross_Vid_Accuracy_STD': cross_vid_accuracy_std})
            
        if per_vid_accuracy_per_class:
            classes = sorted(list(per_vid_accuracy_per_class.values())[0].keys())
            for c in classes:                
                accuracies = np.array([d[c] for d in per_vid_accuracy_per_class.values() if c in d])
                if len(accuracies) > 0:
                    logger.info(f"Cross Vid Class {c} Accuracy Mean: %.4f" % np.mean(accuracies))
                    logger.info(f"Cross Vid Class {c} Accuracy STD : %.4f" % np.std(accuracies))
                    
                    if wandb_prefix:
                        wandb.log({f'{wandb_prefix}_Cross_Vid_Class_{c}_Accuracy_Mean': np.mean(accuracies),
                                f'{wandb_prefix}_Cross_Vid_Class_{c}_Accuracy_STD': np.std(accuracies)})
            
        return acc, edit, cross_vid_accuracy_mean, cross_vid_accuracy_std
        ### end of 
    