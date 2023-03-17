
from src.models.detection.mmdet_base import MMDetModel
from src.models.pose.base import HandPoseResult, PoseModel, PoseResult, TwoHandPoseModel, TwoHandPoseResult

import warnings
import cv2
import numpy as np
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class MMPoseModel(PoseModel):
    """[summary]
    Uses mmdet to detect bounding boxes and mmpose HRNet pretrained model to detect poses.
    Poses are of dimension 21x3 where the third element is the confidence threshold.
    """
    
    def __init__(self, det_model:MMDetModel, pose_config, pose_checkpoint, num_hands=4):
        self.num_hands = num_hands
        self.device = "cuda:0"
        self.det_cat_id = 1
        self.bbox_thr = 0.5
        kpt_thr = 0.3
        radius=4
        thickness = 1
        assert has_mmdet, 'Please install mmdet to run the demo.'

        self.det_model = det_model.get_det_model()
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=self.device.lower())

        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    def forward(self, img, visualize=False):
        # image_name = os.path.join(img_root, img)
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, self.det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = ('backbone',)

        pose_results, returned_outputs = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        # print(len(returned_outputs))
        backbone = returned_outputs[0]['backbone']
        print(backbone.shape)
        """
        pose_results output: list of {bbox, keypoints}, bbox is a list of 4
        every keypoint has a threshold
        pose_results[0]['keypoints'].shape
        (21, 3)
        pose_results[0]['keypoints'][7]
        array([ 59.006683  , 233.56201   ,   0.70138395], dtype=float32)
        
        pose_results[0]['bbox'].shape
        (5,)
        pose_results[0]['bbox']
        array([ 41.366566  , 171.73228   , 148.53111   , 248.97107   ,
            0.99955446], dtype=float32)
            
        returned_outputs[0]['backbone'][0].shape
        (2, 18, 64, 64)
        returned_outputs[0]['backbone'][1].shape
        (2, 36, 32, 32)
        returned_outputs[0]['backbone'][2].shape
        (2, 72, 16, 16)
        returned_outputs[0]['backbone'][3].shape
        (2, 144, 8, 8)
        """
        
        # # show the results
        vis_img = None
        if visualize:
            vis_img = vis_pose_result(
                self.pose_model,
                img,
                pose_results,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=0.3,#kpt_thr,
                radius=4,#radius,
                thickness=1,#thickness,
                show=False)
        
        # res.append(dict(pose_results=pose_results, returned_outputs=returned_outputs))
        # return res
        empty_pose = np.zeros((21, 3))
        empty_bbox = np.zeros(5)
        keypoints = []
        bboxes = []
        pose_results.sort(key=lambda x: x['bbox'][4], reverse=True)
        for i in range(self.num_hands):
            if i <len(pose_results):
                keypoints.append(pose_results[i]['keypoints'])
                bboxes.append(pose_results[i]['bbox'])
            else:
                keypoints.append(empty_pose)
                bboxes.append(empty_bbox)
            
        return PoseResult(keypoints=np.stack(keypoints), 
                          bboxes=np.stack(bboxes),
                          img=vis_img,
                          backbone=backbone)




class TwoHandsMMPoseModel(TwoHandPoseModel):
    """[summary]
    Uses mmdet to detect bounding boxes and mmpose HRNet pretrained model to detect poses.
    Poses are of dimension 21x3 where the third element is the confidence threshold.
    """
    
    def __init__(self, det_model:MMDetModel, pose_config, pose_checkpoint, left_cat=1, right_cat=2):
        self.device = "cuda:0"
        self.left_det_cat_id = left_cat
        self.right_det_cat_id = right_cat
        
        self.bbox_thr = 0.5
        kpt_thr = 0.3
        radius=4
        thickness = 1
        assert has_mmdet, 'Please install mmdet to run the demo.'

        self.det_model = det_model.get_det_model()
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=self.device.lower())

        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.dataset_info = DatasetInfo(self.dataset_info)

    def forward(self, img, visualize=False):
        # image_name = os.path.join(img_root, img)
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(self.det_model, img)

        # keep the person class bounding boxes.
        left_hand_results = process_mmdet_results(mmdet_results, self.left_det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        left_pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            left_hand_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        vis_img = None
        if visualize:
            vis_img = vis_pose_result(
                self.pose_model,
                img,
                left_pose_results,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=0.3,#kpt_thr,
                radius=4,#radius,
                thickness=1,#thickness,
                show=False)

        empty_pose = np.zeros((21, 3))
        empty_bbox = np.zeros(5)
        left_pose_results.sort(key=lambda x: x['bbox'][4], reverse=True)
        if len(left_pose_results) > 0:
            left_hand_pose = HandPoseResult(keypoints=left_pose_results[0]['keypoints'], 
                                            bboxes=left_pose_results[0]['bbox'])
        else:
            left_hand_pose = HandPoseResult(keypoints=empty_pose, bboxes=empty_bbox)

        # keep the person class bounding boxes.
        right_hand_results = process_mmdet_results(mmdet_results, self.right_det_cat_id)

        # test a single image, with a list of bboxes.


        right_pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            right_hand_results,
            bbox_thr=self.bbox_thr,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if visualize:
            vis_img = vis_pose_result(
                self.pose_model,
                vis_img,
                right_pose_results,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=0.3,#kpt_thr,
                radius=4,#radius,
                thickness=1,#thickness,
                show=False)

        right_pose_results.sort(key=lambda x: x['bbox'][4], reverse=True)
        if len(right_pose_results) > 0:
            right_hand_pose = HandPoseResult(keypoints=right_pose_results[0]['keypoints'], 
                                             bboxes=right_pose_results[0]['bbox'])
        else:
            right_hand_pose = HandPoseResult(keypoints=empty_pose, bboxes=empty_bbox)
            
        return TwoHandPoseResult(left=left_hand_pose, right=right_hand_pose, img=vis_img,backbone=None)
