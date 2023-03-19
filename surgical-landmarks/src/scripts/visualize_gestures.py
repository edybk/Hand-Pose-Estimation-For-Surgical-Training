import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1
label_color=(255, 255, 255)
msg_color=(128, 128, 128)


def _white_to_transparency(img):
    img = img.convert("RGBA")
    datas = img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img
    
def _draw_prediction_bars(predictions, ground_truth, actions_dict):
    #draw gestures as time series
    cmap_ = "tab10"
    vmax_ = len(actions_dict)
    gestures_list = [predictions, ground_truth]
    
    fig, axs = plt.subplots(len(gestures_list), 1, figsize=(14,  0.7*len(gestures_list)))
    # axs.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fig.suptitle(f'{""}', fontsize=1)

    for i, gestures in enumerate(gestures_list):
        gestures_ = []

        if "numpy.ndarray" in str(type(gestures)):
            gestures = gestures.tolist()
        for gest in gestures:
            gestures_.append(actions_dict[gest])
        gestures = np.array(gestures_)
        gestures = gestures
        map = np.tile(gestures, (100, 1))
        axs[i].axis('off')
        # axs[i].set_title(names[i], fontsize=10, pad= 10)
        axs[i].imshow(map,cmap=cmap_,aspect="auto",interpolation='nearest',vmax=vmax_)

    plt.tight_layout()
    plt.close(fig)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img

def draw_prediction_bars_on_frame(frame, predictions, ground_truth, actions_dict):
    img = _draw_prediction_bars(predictions, ground_truth, actions_dict)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_frame = pil_frame.convert('RGBA')
    width, height = pil_frame.size
    prediction_bar = Image.fromarray(img)
    prediction_bar = prediction_bar.resize((int(0.7*width), int(0.1*height)))
    prediction_bar = _white_to_transparency(prediction_bar)
    pil_frame.paste(prediction_bar, (int(0.15*width),height-int(0.1*height)), prediction_bar)
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)


def process_video(video_path, results, gt_results, output_root, gesture_to_name, actions_dict):
    actions_dict['G-1']= -1
    reverse_actions_dict = dict([(v, k) for k, v in actions_dict.items()])
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'
    
    os.makedirs(output_root, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        os.path.join(output_root,
                        f'{os.path.basename(video_path)}.gestures.wmv'), fourcc,
        fps, size)

    predictions = ['G-1'] * len(results)
    ground_truth = ['G-1'] * len(gt_results)
    i = 0
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        
        predictions[i] = results[i]
        ground_truth[i] = reverse_actions_dict[gt_results[i].item()]
        
        img = draw_prediction_bars_on_frame(img, predictions, ground_truth, actions_dict)
        
        location = (0, 40)
        text = f"Predicted: {gesture_to_name[results[i]]}"
        
        cv2.putText(img, text, location, FONTFACE, FONTSCALE,
                    label_color, THICKNESS, LINETYPE)
        location = (0, 60)
        text = f"Actual: {gesture_to_name[reverse_actions_dict[gt_results[i].item()]]}"
        cv2.putText(img, text, location, FONTFACE, FONTSCALE,
                    label_color, THICKNESS, LINETYPE)
        
        
        videoWriter.write(img)
        i += 1
            
    
    cap.release()
    videoWriter.release()