scripts/download_trained_model.sh

cd mmpose
python tools/test.py configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/apas/res50_apas_224x224.py work_dirs/res50_apas_224x224/epoch_59.pth