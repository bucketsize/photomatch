import time
import sys
import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray
from PIL import Image
from torch_mtcnn import detect_faces
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from torch.utils.data import DataLoader
from torchvision import datasets
from face_store import FaceStore
from face_utils import in_box, in_range, print_nice, get_image_files
import torch

print("torch", torch.__version__)
print("torch.vulkan", torch.is_vulkan_available())

# lazy, eager
def get_image(image_path):
    return (Image.open(image_path), plt.imread(image_path), image_path)

def show_image(image_path):
    plt.imshow(plt.imread(image_path))

def show_overlay(image, roi_boxes):
    plt.imshow(image)
    ax = plt.gca()

    for roi in roi_boxes:
        x1, y1, x2, y2 = roi["box"]
        roi_border = Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color='red')
        ax.add_patch(roi_border)
    plt.show()

vfint = np.vectorize(int)

def get_images(image_inf, rois, confidence=0.85, required_size=(224, 224)):
    iinfo,image,name = image_inf
    images = []

    for roi in rois:
        x1, y1, x2, y2 = roi["box"] 
        # print("roi: ", x1, y1, x2, y2)
        c = roi["confidence"]
        w, h = iinfo.size
        if in_box((0,0,w,h), x1, y1) and in_box((0,0,w,h), x2, y2): 
            bb = image[y1:y2, x1:x2]
            img = Image.fromarray(bb).resize(required_size)
            img_array = asarray(img)
            images.append(img_array)
        else:
            print("drop invalid roi [%s]" % (name), roi["box"])

    return images

class FaceUsurper:
    def __init__(self, image_size=160):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        # :default:
        # image_size=160, margin=0, min_face_size=20,
        # thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        # select_largest=True, selection_method=None, keep_all=False, device=None
        self.mtcnn = MTCNN(
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        if len(sys.argv) < 2:
            raise "invalid arg; expect 1 (db_f)"
        self.db_f = sys.argv[1]
        self.store = FaceStore(self.db_f)
        self.embeddings = []
        self.faces = []
        self.matches = []

    def get_faces(self, image_inf, required_size=(224, 224)):
        image,_,name = image_inf
        image_size,_ = required_size
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        faces_list = []
        for i, (box, prob, point) in enumerate(zip(boxes, probs, points)):
            face_path = "/var/tmp/{}/face_{}_{}.png".format(
                self.db_f.replace("/", "_"),
                name.replace("/", "_"),
                i)
            ftensor = extract_face(
                image, box,
                save_path=face_path)
            faces_list.append({
                "image_path": name,
                "face_path": face_path,
                "box": vfint(box),
                "confidence": prob,
                "tensor": ftensor
            })
        # print("faces [%s] = [%d]" % (name, len(faces_list)))
        return faces_list

    def compare_matches(self, faces, es):
        match_list = []
        for i, ei in enumerate(es):
            for j, ej in enumerate(self.embeddings):
                score = (ei - ej).norm().item()
                if 0 < score and score < 0.3:
                    # print("> ", (i,j)," / ", len(self.faces), len(self.embeddings))
                    s = faces[i]
                    t = self.faces[j]
                    match_list.append((s["face_path"], t["face_path"], score))
        print("matched c ~ (0, 0.3) = [%d]" % len(match_list))            
        return match_list

    def match_faces(self, faces):
        print("match_faces started ...")
        if len(faces) == 0:
            print("expect non empty list of faces.tensor")
            return [] 
        aligned0 = list(map(lambda f: f["tensor"], faces))
        aligned1 = torch.stack(aligned0).to(self.device)
        embeddings = self.resnet(aligned1).detach().cpu()
        return embeddings

    def extract_faces(self, image_path):
        image_info = get_image(image_path)
        faces = self.get_faces(image_info)
        return faces

    def process(self, image_path):
        print("started [%s]" % image_path)
        if len(self.store.find_image_path(image_path)) == 0:
            faces = self.extract_faces(image_path)
            print("detected = [%d] " % len(faces))
            return faces
        else:
            print("duplicate")
            return []

    def update(self, faces=[], embeddings=[], matches=[], r_matches=[]):
        if len(faces) > 0:
            self.faces += faces
            self.store.save_faces(faces)
        if len(embeddings) > 0:
            self.embeddings += embeddings
        if len(matches) > 0:
            self.matches += matches
            self.store.save_matches(matches)
           
    def stats(self):
        return {
            "faces": len(self.faces),
            "embeddings": len(self.embeddings),
            "matches": len(self.matches)
        }

def main():
    fsup = FaceUsurper()
    for image_path in get_image_files("test"):
        faces = fsup.process(image_path)
        fsup.update(faces=faces)

        ems = fsup.match_faces(faces)
        fsup.update(embeddings=ems)
       
        cms = fsup.compare_matches(faces, ems)
        fsup.update(matches=cms)
        
    print_nice(fsup.stats())

main()
