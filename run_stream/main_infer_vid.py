import time
import sys
import subprocess as sp
import cv2
import numpy as np
import os
import math
from multiprocessing import Process
from flask import Flask, render_template, Response
from threading import Thread
import json
import tempfile
from contextlib import redirect_stdout
import torch
import torch.nn.functional as F
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), '../retinanet'))
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)

from retinanet.data import DataIterator, RotatedDataIterator
from retinanet.dali import DaliDataIterator
from retinanet.model import Model
from retinanet.utils import Profiler, rotate_box
from retinanet import utils
from retinanet._C import Engine

# Moodoong RTSP source from XNX1
RTSP_ADDR = "rtsp://59.3.71.98:8554/main"
MODEL_PATH = "./pretrained_weigths/ResNet18FPN_cow.pth"

# Flask initialize
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

def odtk_infer():
    global img_out
    img_out = np.zeros((720,1280,3),dtype='uint8')
    #Initial parameters
    batch_size=1
    mixed_precision=True
    rotated_bbox=True

    resize = 640
    max_size = 640
    mean_fix = [0.485, 0.456, 0.406]
    std_fix = [0.229, 0.224, 0.225]
    ratios = 0.5 #1280x720 -> 640x360
    det_thr = 0.5
    backend = 'pytorch'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    this_det = {}

    #Initialize model
    model = load_model(MODEL_PATH, rotated_bbox)
    if model: model.share_memory()

    # 'Run inference on images from video stream'
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'
    stride = model.module.stride if isinstance(model, DDP) else model.stride

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare model
    if backend is 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if torch.cuda.is_available(): model = model.cuda()
        model = amp.initialize(model, None,
                               opt_level='O2' if mixed_precision else 'O0',
                               keep_batchnorm_fp32=True,
                               verbosity=0)
        model.eval()  




    video_on = cv2.VideoCapture(RTSP_ADDR) #Input video stream
    with torch.no_grad():
        while True:
            
            _, img = video_on.read()
            # Convert to tensor and normalize
            img_r = cv2.resize(img, dsize=(640, 352), interpolation=cv2.INTER_AREA)
            img_r = img_r.astype('float32') / 255.0
            img_t = torch.from_numpy(img_r)
            img_t = img_t.permute(2, 0, 1)    

            for t, mean, std in zip(img_t, mean_fix, std_fix):
                t.sub_(mean).div_(std)

            # Apply padding
            pw, ph = ((stride - d % stride) % stride for d in img_r.shape[:-1])
            img_t = F.pad(img_t, (0, pw, 0, ph))
            img_t = torch.reshape(img_t, (1,img_t.shape[0],img_t.shape[1],img_t.shape[2]))
            if torch.cuda.is_available():
                img_t = img_t.cuda(non_blocking=True)

            # Forward pass
            results = model(img_t, rotated_bbox) #Need to add model size (B, 3, W, H)
            # Copy buffers back to host
            scores, boxes, classes = [r.cpu() for r in results]

            keep = (scores > det_thr)
            scores = scores[keep].view(-1)
            if rotated_bbox:
                boxes = boxes[keep, :].view(-1, 6)
                boxes[:, :4] /= ratios
            else:
                boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()

            dets = []
            img_out = img
            for score, box, cat in zip(scores, boxes, classes):
                if rotated_bbox:
                    x1, y1, x2, y2, sin, cos = box.data.tolist()
                    theta = np.arctan2(sin, cos)
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    seg = rotate_box([x1, y1, w, h, theta])
                else:
                    x1, y1, x2, y2 = box.data.tolist()
                this_det['category_id'] = cat.item()
                
                if rotated_bbox:
                    this_det['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1, theta]
                    this_det['segmentation'] = [seg]
                else:
                    this_det['bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                
                dets.append(this_det)

                img_out = annot_overlay(img_out, dets)


def annot_overlay(img, dets):
    category_dic={0:'unknown',1:'standing',2:'sternallying',3:'laterallying',4:'mounting',5:'sitting'} #class name
    category_color={0:(128,128,64),1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,0,255),5:(255,255,0)} #class color

    for det in dets:
            
        category_id=det['category_id']
        category_id+=1
        category_name=category_dic[category_id]
                                        #[bbox]
        x_min=det['bbox'][0]    #x
        y_min=det['bbox'][1]    #y
        width=det['bbox'][2]    #width
        height=det['bbox'][3]   #height
        radian=det['bbox'][4]   #theta
                                                #[segmentation]
        x1=det['segmentation'][0][0]    #x1
        y1=det['segmentation'][0][1]    #y1
        x2=det['segmentation'][0][2]    #x2
        y2=det['segmentation'][0][3]    #y2
        x3=det['segmentation'][0][4]    #x3
        y3=det['segmentation'][0][5]    #y3
        x4=det['segmentation'][0][6]    #x4
        y4=det['segmentation'][0][7]    #y4
        cx=round(x_min+(width/2))   #center x
        cy=round(y_min+(height/2))  #center y
        #json data bbox의 x,y,w,h 를 가지고 x1,y1 x2,y2 x3,x4 변환 후  -radian 회전 
        rotated_x1,rotated_y1=rotate((cx,cy),(x_min,y_min),-radian)
        rotated_x2,rotated_y2=rotate((cx,cy),(x_min,y_min+height),-radian)
        rotated_x3,rotated_y3=rotate((cx,cy),(x_min+width,y_min+height),-radian)
        rotated_x4,rotated_y4=rotate((cx,cy),(x_min+width,y_min),-radian)
        #draw 
        img=cv2.line(img,(round(rotated_x1),round(rotated_y1)),(round(rotated_x2),round(rotated_y2)),category_color[category_id],2)
        img=cv2.line(img,(round(rotated_x2),round(rotated_y2)),(round(rotated_x3),round(rotated_y3)),category_color[category_id],2)
        img=cv2.line(img,(round(rotated_x3),round(rotated_y3)),(round(rotated_x4),round(rotated_y4)),category_color[category_id],2)
        img=cv2.line(img,(round(rotated_x4),round(rotated_y4)),(round(rotated_x1),round(rotated_y1)),category_color[category_id],2)
        
        cv2.putText(img, category_name, (round(cx-10),round(cy-10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, category_color[category_id], 2)

    return img


def rotate(origin, point, angle): # origin을 중심으로 point를 angle(radian) 한 값이 (qx,qy) 

    ox, oy = origin 
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def load_model(MODEL_PATH, rotated_bbox):
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError('Model file {} does not exist!'.format(MODEL_PATH))

    model = None
    state = {}
    _, ext = os.path.splitext(MODEL_PATH)

    if ext == '.pth' or ext == '.torch':
        print('Loading model from {}...'.format(os.path.basename(MODEL_PATH)))
        model, _ = Model.load(filename=MODEL_PATH, rotated_bbox=rotated_bbox)
        print(model)

    elif ext in ['.engine', '.plan']:
        print('Loading CUDA engine from {}...'.format(os.path.basename(MODEL_PATH)))
        model = Engine.load(MODEL_PATH)

    else:
        raise RuntimeError('Invalid model format "{}"!'.format(ext))

    return model


def threadVid():
    
    while True:
        _, jpeg = cv2.imencode('.jpg', img_out)
        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                        
#Flask thread run
@app.route('/web_feed')
def video_feed():
    return Response(threadVid(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')	

def main():
  
    th_vid_tmp = Thread(target= odtk_infer, args=())
    th_vid_tmp.start()    

    app.run(host='0.0.0.0', debug=False)

if __name__ == "__main__":
    main()