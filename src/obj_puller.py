# https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/#download-the-code

import sys
import os
import pickle
import numpy as np
import cv2
import torch
import argparse
from torchvision import models 
from torchvision.models import detection
from img_utils import get_image_files, cv_to_torch
from obj_store import ObjStore

CONFIDENCE = 0.55
IN_LABELS = "coco_labels.txt"
MODELS = {
	"frcnn-resnet": {
        "fn": detection.fasterrcnn_resnet50_fpn,
        "wt": detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        "wt_b": models.ResNet50_Weights.IMAGENET1K_V1
    },
	"frcnn-mobilenet": {
        "fn": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        "wt": detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
        "wt_b": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    },
	"retinanet": {
        "fn": detection.retinanet_resnet50_fpn,
        "wt": detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1,
        "wt_b": models.ResNet50_Weights.IMAGENET1K_V1
    }
}

class ObjPuller:
    def __init__(self, modeld, db_f):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('ii running on device: {}'.format(self.device))
        self.tags = dict(enumerate(open(IN_LABELS)))
        print(modeld)
        pt_model = modeld["fn"](
            weights=modeld["wt"],
            weights_backbone=modeld["wt_b"],
            progress=True,
            num_classes=len(self.tags),
            pretrained_backbone=True)
        self.model = pt_model.eval().to(self.device)
        self.store = ObjStore(db_f)

    def load_image(self, image_path):
        image = cv_to_torch(cv2.imread(image_path))
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        return image

    def pull(self, image_path):
        print("ii %s" % image_path )
        image = self.load_image(image_path)
        detects = self.model(image)[0]
        objs, ts = [], []
        for i in range(0, len(detects["boxes"])): 
            confidence = detects["scores"][i]
            if confidence > CONFIDENCE:
                idx = int(detects["labels"][i])-1
                print(" - {:.2f}% {}".format(confidence * 100, self.tags[idx]))
                box = detects["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                objs.append((None, [startX, startY, endX, endY], confidence, [], idx, "COCO", self.tags[idx]))
        self.store.save_objects(image_path, objs, ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str,
        default="retinanet",
        choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
        help="name of the object detection model")
    ap.add_argument("-c", "--confidence", type=float,
        default=0.57,
        help="minimum probability to filter weak detections")
    ap.add_argument("-p", "--path", type=str,
        required=True,
        help="minimum probability to filter weak detections")
    ap.add_argument("-d", "--db", type=str,
        required=True,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    pm = MODELS[args["model"]]
    cn = ObjPuller(pm, args["db"])
    for image_path in get_image_files(args["path"]):
        if len(cn.store.find_image_path(image_path)) > 0:
            print("ii dup "+image_path)
        else:
            try:
                cn.pull(image_path)
            except Exception as e:
                print("ii skip on error: %s" % image_path, e)
                pass

main()
