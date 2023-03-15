from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector

pose_config = 'mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/res50_apas_224x224.py'
pose_checkpoint = 'mmpose/work_dirs/res50_apas_224x224/epoch_59.pth'
det_config = '../detection/mmdetection/configs/yolox/yolox_s_8x8_300e_apas_allclass.py'
det_checkpoint = '../detection/mmdetection/work_dirs/yolox_s_8x8_300e_apas_allclass/epoch_690.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
det_model = init_detector(det_config, det_checkpoint)

img = 'mmpose/data/apas/images/P020_balloon1_4235.jpg'

# inference detection
mmdet_results = inference_detector(det_model, img)

# extract right hand bounding boxes from the detection results
hand1_results = process_mmdet_results(mmdet_results, cat_id=1)

# extract left hand bounding boxes from the detection results
hand2_results = process_mmdet_results(mmdet_results, cat_id=2)

# inference pose
pose_results, returned_outputs = inference_top_down_pose_model(
    pose_model,
    img,
    hand1_results+hand2_results,
    bbox_thr=0.3,
    format='xyxy',
    dataset='TopDownCocoDataset')

vis_result = vis_pose_result(
    pose_model,
    img,
    pose_results,#pose_results1+pose_results2,
    kpt_score_thr=0.,
    dataset='TopDownCocoDataset',
    show=False)


# reduce image size
import cv2
vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

import tempfile
import os.path as osp
with tempfile.TemporaryDirectory() as tmpdir:
    file_name = osp.join(tmpdir, 'pose_results.png')
    # cv2.imwrite(file_name, vis_result)
    cv2.imwrite('pose_results.png', vis_result)
