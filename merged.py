import datetime
import pyttsx3
import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from gaze_tracking import GazeTracking

gaze = GazeTracking()

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

classes_to_filter = ['train']  # You can give list of classes to filter by name ['train','person']

opt = {
    "weights": "G:\HCI\Project codes\YOLOV7\weights\yolov7.pt",  # Path to weights file default weights are for nano model
    "yaml": "data/coco.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Video information
fps = 30  # Adjust the frame rate as needed
w, h = 640, 480  # Set the desired width and height for the video output
gaze_timer = 0
yolo_timer = 0

# Initializing video object
video = cv2.VideoCapture(1)  # Use index 0 for the default camera

# Initializing object for writing video output
output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

# Initializing model and setting it for inference
with torch.no_grad():
    weights = 'G:\HCI\Project codes\YOLOV7\weights\yolov7.pt'  # Replace with the path to your YOLOv7 weights
    imgsz = 416  # Set the input image size for the model
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    while True:
        _, frame = video.read()

        gaze.refresh(frame)

        new_frame = gaze.annotated_frame()
        text = ""

        if gaze.is_right():
            text = "Looking right"
            engine.say('someone is looking to you')
            engine.runAndWait()
        elif gaze.is_left():
            text = "Looking left"
            engine.say('someone is looking to you')
            engine.runAndWait()
        elif gaze.is_center():
            text = "Looking center"
            engine.say('someone is looking to you')
            engine.runAndWait()
            print('someone is looking to you')

        ret, img0 = video.read()

        if ret:
            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.6)

            t2 = time_synchronized()
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

                        # Speak the name of the detected object
                        engine.say(label)
                        engine.runAndWait()

            cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
            cv2.putText(img0, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
            cv2.imshow("Object Detection and Gaze Tracking", np.hstack((img0, new_frame)))
            output.write(cv2.resize(img0, (w, h)))

            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit+
                break
            #print(gaze_timer)
            gaze_timer = gaze_timer + 1
        else:
            break

output.release()
video.release()
cv2.destroyAllWindows()