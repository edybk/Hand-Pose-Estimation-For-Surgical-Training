import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = 'mmdetection/configs/yolox/yolox_s_8x8_300e_apas_allclass.py'
# Setup a checkpoint file to load
checkpoint = 'mmdetection/work_dirs/yolox_s_8x8_300e_apas_allclass/epoch_690.pth'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

# Use the detector to do inference
img = 'mmdetection/data/apas_allclass/coco/images/P020_balloon1_4235.jpg'
result = inference_detector(model, img)
# Let's plot the result
model.show_result(img, result, score_thr=0.75, out_file = "detection_results.png")