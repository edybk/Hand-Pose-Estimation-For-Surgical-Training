import argparse
import os
import numpy as np
from src.scripts.download_proxy_exports import download_proxy_exports
from src.dataset.proxy import HandOrientationProxy, HandVelocityProxy, KPDistanceProxy, KPVelocityProxy
from src.dataset.utils import all_gestures, end2end, end2end_clips_export, load_all_samples, load_raw_all_combined, plot_bars, plot_box_plots

use_downloaded = False

hand_orientation_proxes = [
    HandOrientationProxy('right'),
    HandOrientationProxy('left')
]
kp_distance_proxies = [
    KPDistanceProxy("right", 4, 8),
    KPDistanceProxy("left", 4, 8)
]
kp_velocity_proxies = [
    KPVelocityProxy("right", 4),
    KPVelocityProxy("right", 8),
    KPVelocityProxy("right", 12)
]

all_proxies = hand_orientation_proxes + kp_distance_proxies + kp_velocity_proxies + [HandVelocityProxy("right")]


videos_dir = os.getenv("FRONTAL_VIDEOS")
repo_root = os.getenv("REPOSITORY_ROOT")

data_root_dir = os.path.join(repo_root, "surgical-landmarks", "data")
downloaded_exports_root_dir = os.path.join(data_root_dir, "downloaded", "proxy_exports")

dataset_root_dir = os.path.join(data_root_dir, "apas_tcn_v2")

generated_root_dir = os.path.join(repo_root, "surgical-landmarks", "data", "generated")
combined_centers_path = os.path.join(generated_root_dir, "poses_detections_combined_centers_features") #TODO: populate from drive
raw_combined_centers_path = os.path.join(generated_root_dir, "poses_detections_combined_features") #TODO: populate from drive
all_combined = load_all_samples(combined_centers_path)
raw_all_combined = load_raw_all_combined(raw_combined_centers_path)


def get_proxy_export_out_root(use_downloaded):
    return os.path.join(generated_root_dir, "proxy_exports") if not use_downloaded else downloaded_exports_root_dir


def visualize_proxies():
    sample_name = "P016_balloon1"

    # hand orientation
    end2end(
            dataset_root_dir=dataset_root_dir,
            videos_dir=videos_dir,
            all_combined=all_combined,
            raw_all_combined=raw_all_combined,
            sample_name=sample_name,
            kp_clean_method="last", 
            kp_filter_method="savgol", 
            tool_clean_method=None, 
            tool_filter_method=None, 
            proxy=hand_orientation_proxes[0],
            test=30
        )
    
    # kp distance
    end2end(
            dataset_root_dir=dataset_root_dir,
            videos_dir=videos_dir,
            all_combined=all_combined,
            raw_all_combined=raw_all_combined,
            sample_name=sample_name,
            kp_clean_method="last", 
            kp_filter_method="savgol", 
            tool_clean_method=None, 
            tool_filter_method=None, 
            proxy=kp_distance_proxies[0],
            test=399
        )
    
    # kp velocity
    end2end(
            dataset_root_dir=dataset_root_dir,
            videos_dir=videos_dir,
            all_combined=all_combined,
            raw_all_combined=raw_all_combined,
            sample_name=sample_name,
            kp_clean_method="last", 
            kp_filter_method="savgol", 
            tool_clean_method=None, 
            tool_filter_method=None, 
            proxy=kp_velocity_proxies[1],
            test=1650
        )

def export_proxies():
    clip_bys = ["gestures"] #["tool_usage_left", "tool_usage_right"]

    out_dir_root = get_proxy_export_out_root(False)

    for sample_name in all_combined:
        sample_base_name = os.path.basename(sample_name).split(".")[0]
        for p in all_proxies:
            for clip_by in clip_bys:
                print(f"exporting {sample_base_name} {clip_by} clips for proxy {p.name()}", flush=True)
                end2end_clips_export(
                    dataset_root_dir,
                    videos_dir,
                    all_combined,
                    raw_all_combined,
                    sample_base_name, 
                    kp_clean_method="last", 
                    kp_filter_method="savgol", 
                    tool_clean_method=None, 
                    tool_filter_method=None, 
                    proxy=p,
                    out_dir=os.path.join(out_dir_root, p.name()),
                    skip_no_gesture=True, #False # only for duration_proxies!,
                    clip_by=clip_by
                )
            print(f"done {sample_base_name} - {p.name()}", flush=True)
        print(f"done {sample_base_name}", flush=True)  
        
    print("finished!", flush=True)

    
def plot_novice_vs_expert(use_downloaded=False):
    proxy_export_dir = get_proxy_export_out_root(use_downloaded)
    # print(f"proxy export dir: {proxy_export_dir}")
    if not os.path.exists(proxy_export_dir):
        if use_downloaded:
            download_proxy_exports()
        else:
            raise "You need to export the proxy values first or set use_downloaded to True"

    for proxy in all_proxies:
        plot_bars(all_gestures(dataset_root_dir), proxy=proxy.name(), export_dir=proxy_export_dir, save=True, aggregate=np.mean, max_p_value = 0.05)
        plot_box_plots(all_gestures(dataset_root_dir), proxy=proxy.name(), export_dir=proxy_export_dir, save=True)

    # break


parser = argparse.ArgumentParser(
                    prog='Surgical Skill Proxy Tool',
                    description='.',
                    epilog='')

parser.add_argument('action', type=str, choices=["visualize", "export", "plot"])
parser.add_argument('--download', dest='use_downloaded', action='store_true')
parser.add_argument('--no-download', dest='use_downloaded', action='store_false')
parser.set_defaults(use_downloaded=True)

args = parser.parse_args()

if args.action == "visualize":
    visualize_proxies()
elif args.action == "export":
    export_proxies()
elif args.action == "plot":
    plot_novice_vs_expert(use_downloaded=args.use_downloaded)
else:
    raise "invalid action"
