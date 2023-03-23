
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2




def x_plot_proxy(values, axes_visible=False):
    fig, ax = plt.subplots(1, 1)
    plt.plot(list(range(len(values))), values)
    # X-axis tick label
    # plt.xticks(color='w')
    ax.get_xaxis().set_visible(axes_visible)
    # Y-axis tick label
    # plt.yticks(color='w')
    ax.get_yaxis().set_visible(axes_visible)
    plt.tight_layout()
    plt.close(fig)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


def x_plot_proxy_on_frame(values, frame, position=1):
    img = x_plot_proxy(values)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_frame = pil_frame.convert('RGBA')
    width, height = pil_frame.size
    prediction_bar = Image.fromarray(img)
    prediction_bar = prediction_bar.resize((int(0.7*width), int(0.1*height)))
    # prediction_bar = _white_to_transparency(prediction_bar)
    # pil_frame.paste(prediction_bar, (int(0.15*width),height-int(0.1*height)), prediction_bar)
    pil_frame.paste(prediction_bar, (int(0.15*width), height-int(position*int(0.1*height))))
    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

class Proxy:

    # abstract
    def calc(self, package):
        pass

    def draw(self, frame, package, vid_proxy_vals, frame_num, position=1):
        proxy_val = self.calc(package)
        vid_proxy_vals[frame_num] = proxy_val
        ret = x_plot_proxy_on_frame(vid_proxy_vals, frame, position)
        ret = self.draw_proxy(frame, package)
        return ret

    def calculate(self, package, frame_num, vid_proxy_vals):
        proxy_val = self.calc(package)
        vid_proxy_vals[frame_num] = proxy_val
        
    
    def name(self):
        return type(self).__name__
    
    def draw_proxy(self, frame, package):
        return frame




class HandOrientationProxy(Proxy):
    
    def __init__(self, hand):
        assert(hand == "left" or hand == 'right')
        self.hand = hand
        
    def heuristic(self, poses):
        knucle1 = poses[5]
        knucle2 = poses[17]
        if self.hand == 'right':
            return knucle1[0]-knucle2[0]
        else:
            return knucle2[0]-knucle1[0]
    
    
    def get_poses_from_package(self, package):
        if self.hand == 'right':
            poses = package["right_keypoints"]
        else:
            poses= package["left_keypoints"]
        return poses
        
    def calc(self, package):
        return self.heuristic(self.get_poses_from_package(package))
    
    def name(self):
        return type(self).__name__ + "_" + self.hand
    
    def draw_proxy(self, frame, package):
        poses = self.get_poses_from_package(package)
        point1 = poses[5][:2].astype(int)
        point2 = poses[17][:2].astype(int)

        frame = cv2.line(frame, point1, point2, color=(0, 0, 255), thickness=3)
        frame = cv2.putText(frame, text=self.name() + ": " + str("{:.2f}".format(self.heuristic(poses))), org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1, color=(255, 0, 0), thickness=2)
        return frame
  

def distance(x1, y1, x2, y2):
    return ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)


def velocity(center_history):
    frame_cnt = 30

    if len(center_history) < frame_cnt:
        return 0
    # print(len(center_history))
    center_history = center_history[-frame_cnt:]

    total_distance = 0
    for i in range(1, min(len(center_history), frame_cnt)):
        a, b = center_history[i-1], center_history[i]
        total_distance += distance(a[0], a[1], b[0], b[1])
    velocity = total_distance / frame_cnt
    return velocity


class KPDistanceProxy(Proxy):

    def __init__(self, hand, kp_index1, kp_index2):
        assert(kp_index1 >= 0 and kp_index1 < 21)
        assert(kp_index2 >= 0 and kp_index2 < 21)
        self.kp_index1 = kp_index1
        self.kp_index2 = kp_index2
        assert(hand == "left" or hand == 'right')
        self.hand = hand

    def get_centers(self, package):
        center1 = package["left_keypoints"][self.kp_index1][:2] if self.hand == "left" else package["right_keypoints"][self.kp_index1][:2]
        center2 = package["left_keypoints"][self.kp_index2][:2] if self.hand == "left" else package["right_keypoints"][self.kp_index2][:2]
        return center1, center2

    def calc(self, package):
        center1, center2 = self.get_centers(package)
        
        return distance(center1[0], center1[1],
                 center2[0], center2[1])
        
        
    def draw_proxy(self, frame, package):
        
        center1, center2 = self.get_centers(package)
        point1 = center1[:2].astype(int)
        point2 = center2[:2].astype(int)
        
        frame = cv2.line(frame, point1, point2, color=(0, 0, 255), thickness=3)
        frame = cv2.putText(frame, text=self.name() + ": " + str("{:.2f}".format(self.calc(package))), org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1, color=(255, 0, 0), thickness=2)
        return frame


    def name(self):
        return type(self).__name__ + "_" + self.hand + "_" + str(self.kp_index1) + "_" + str(self.kp_index2)




class KPVelocityProxy(Proxy):

    def __init__(self, hand, kp_index):
        assert(kp_index >= 0 and kp_index < 21)
        self.kp_index = kp_index
        assert(hand == "left" or hand == 'right')
        self.hand = hand

    def get_kp_history(self, package):
        return package["histories"][f"{self.hand}_{self.kp_index}"]

    def calc(self, package):
        val = package["left_keypoints"][self.kp_index, :2] if self.hand == "left" else package["right_keypoints"][self.kp_index, :2]
        package["histories"][f"{self.hand}_{self.kp_index}"].append(val)
        kp_velocity = velocity(self.get_kp_history(package))
        return kp_velocity


    def name(self):
        return type(self).__name__ + "_" + self.hand + "_" + str(self.kp_index)

    def draw_proxy(self, frame, package):
        val = self.calc(package)
        kp_history = self.get_kp_history(package)[-30:]
        for i, kp in enumerate(kp_history):
            frame = cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=1, color=(0, 0, 255), thickness=1)
        
        frame = cv2.putText(frame, text=self.name() + ": " + str("{:.2f}".format(val)), org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1, color=(255, 0, 0), thickness=2)
        return frame


class HandVelocityProxy(Proxy):

    def __init__(self, hand):
        assert(hand == "left" or hand == 'right')
        self.hand = hand

    def get_history(self, package):
        return package["histories"][self.hand]

    def calc(self, package):
        
        val = package["left_center"] if self.hand == "left" else package["right_center"]
        package["histories"][self.hand].append(val)
        hand_velocity = velocity(self.get_history(package))
        return hand_velocity

    def name(self):
        return type(self).__name__ + "_" + self.hand

    def draw_proxy(self, frame, package):
        val = self.calc(package)
        kp_history = self.get_history(package)[-30:]
        for i, kp in enumerate(kp_history):
            frame = cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=1, color=(0, 0, 255), thickness=1)
        
        frame = cv2.putText(frame, text=self.name() + ": " + str("{:.2f}".format(val)), org=(25, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1, color=(255, 0, 0), thickness=2)
        return frame