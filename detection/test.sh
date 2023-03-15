scripts/download_trained_model.sh

cd mmdetection
python tools/test.py configs/yolox/yolox_s_8x8_300e_apas_allclass.py work_dirs/yolox_s_8x8_300e_apas_allclass/epoch_690.pth --eval bbox