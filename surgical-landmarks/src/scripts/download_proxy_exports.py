import os
import gdown
import zipfile

def download_proxy_exports():

    # download exports
    repo_root = os.getenv("REPOSITORY_ROOT")
    # generated_root_dir = os.path.join(repo_root, "surgical-landmarks", "data", "generated")
    # proxy_export_out_root = os.path.join(generated_root_dir, "proxy_exports")

    data_root_dir = os.path.join(repo_root, "surgical-landmarks", "data")
    downloaded_root_dir = os.path.join(data_root_dir, "downloaded")
    os.makedirs(downloaded_root_dir, exist_ok=True)

    fild_id = "1BM6DlybSxl1588cYHxnFfgoxYyTWjKmh"
    proxy_exports_zip_path = os.path.join(downloaded_root_dir, "proxy_exports.zip")
    gdown.download(id=fild_id, output=proxy_exports_zip_path, quiet=False)

    # unzip
    with zipfile.ZipFile(proxy_exports_zip_path,"r") as zip_ref:
        zip_ref.extractall(downloaded_root_dir)

    os.remove(proxy_exports_zip_path)

    

    
