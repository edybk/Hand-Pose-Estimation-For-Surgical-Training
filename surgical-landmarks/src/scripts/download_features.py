import os
import gdown
import zipfile

def prepare_features(download_location, data_root):
    apas_tcn_v2_path = os.path.join(data_root, "apas_tcn_v2")


    # create download directory
    os.makedirs(download_location, exist_ok=True)

    # download keypoints features

    keypoints_features_root = os.path.join(download_location, "keypoints_features")
    os.makedirs(keypoints_features_root, exist_ok=True)

    keypoints_frontal_zip_id = "1Ah_VKUlvhtf0yd2UXLISAz-BiqEd6-CK"
    keypoints_frontal_zip_path = os.path.join(keypoints_features_root, "smooth_final.zip")
    gdown.download(id=keypoints_frontal_zip_id, output=keypoints_frontal_zip_path, quiet=False)

    with zipfile.ZipFile(keypoints_frontal_zip_path,"r") as zip_ref:
        zip_ref.extractall(keypoints_features_root)

    os.remove(keypoints_frontal_zip_path)
        
    os.symlink(os.path.join(keypoints_features_root, "smooth_final"), os.path.join(apas_tcn_v2_path, "smooth_final"), target_is_directory=True)


    keypoints_closeup_zip_id = "1aIAgKuqbYeC85JNRqIn9bnAhR2VHq435"
    keypoints_closeup_zip_path = os.path.join(keypoints_features_root, "smooth_final_closeup.zip")
    gdown.download(id=keypoints_closeup_zip_id, output=keypoints_closeup_zip_path, quiet=False)

    with zipfile.ZipFile(keypoints_closeup_zip_path,"r") as zip_ref:
        zip_ref.extractall(keypoints_features_root)

    os.remove(keypoints_closeup_zip_path)
        
    os.symlink(os.path.join(keypoints_features_root, "smooth_final_closeup"), os.path.join(apas_tcn_v2_path, "smooth_final_closeup"), target_is_directory=True)



    # download i3d features


    i3d_features_root = os.path.join(download_location, "i3d_features")
    os.makedirs(i3d_features_root, exist_ok=True)

    # frontal
    frontal_zip_id = "1b6XswZxA2roGV230g1PP9oT3aVln7znO"
    frontal_zip_path = os.path.join(i3d_features_root, "frontal.zip")
    gdown.download(id=frontal_zip_id, output=frontal_zip_path, quiet=False)

    with zipfile.ZipFile(frontal_zip_path,"r") as zip_ref:
        zip_ref.extractall(i3d_features_root)

    os.remove(frontal_zip_path)

    apas_tcn_v2_path = os.path.join(data_root, "apas_tcn_v2")
    os.symlink(os.path.join(i3d_features_root, "frontal", "i3d"), os.path.join(apas_tcn_v2_path, "i3d_frontal"), target_is_directory=True)

    # closeup
    closeup_zip_id = "1wgBazc8bxy22iEySaTmdl8uwa4koMdPj"
    closeup_zip_path = os.path.join(i3d_features_root, "closeup.zip")
    gdown.download(id=closeup_zip_id, output=closeup_zip_path, quiet=False)


    with zipfile.ZipFile(closeup_zip_path,"r") as zip_ref:
        zip_ref.extractall(i3d_features_root)

    os.remove(closeup_zip_path)

    os.symlink(os.path.join(i3d_features_root, "closeup", "i3d"), os.path.join(apas_tcn_v2_path, "i3d_closeup"), target_is_directory=True)


    # both views
    both_zip_id = "1bLbJ1B0Z5BHIOY1nzPxhCHTX2sWTAqJq"
    both_zip_path = os.path.join(i3d_features_root, "both.zip")
    gdown.download(id=both_zip_id, output=both_zip_path, quiet=False)

    with zipfile.ZipFile(both_zip_path,"r") as zip_ref:
        zip_ref.extractall(i3d_features_root)
        
    os.remove(both_zip_path)

    os.symlink(os.path.join(i3d_features_root, "both", "i3d"), os.path.join(apas_tcn_v2_path, "i3d_both"), target_is_directory=True)


    
