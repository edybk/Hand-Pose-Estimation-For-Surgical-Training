import os
import gdown
import zipfile

def prepare_dataset(data_root, smooth_root):
    # create directory
    os.makedirs(data_root, exist_ok=True)

    # download dataset
    fild_id = "1D61xUstQrpQU96ExH0GO08FQnt4TkCmt"
    apas_tcn_v2_zip_path = os.path.join(data_root, "apas_tcn_v2.zip")
    gdown.download(id=fild_id, output=apas_tcn_v2_zip_path, quiet=False)

    # unzip
    with zipfile.ZipFile(apas_tcn_v2_zip_path,"r") as zip_ref:
        zip_ref.extractall(data_root)

    os.remove(apas_tcn_v2_zip_path)

    apas_tcn_v2_path = os.path.join(data_root, "apas_tcn_v2")
    os.symlink(smooth_root, os.path.join(apas_tcn_v2_path, "smooth_final"))
    
