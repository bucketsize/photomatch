import time
import sys
import os
import torch
from torch import nn
from torch import linalg as LA
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
    def get_uid(self):
        self.count += 1
        return "%05d" % self.count

    def __init__(self, db_f, images_path, image_size=160):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        # :default:
        # image_size=160, margin=0, min_face_size=20,
        # thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        # select_largest=True, selection_method=None, keep_all=False, device=None
        self.mtcnn = MTCNN(
            margin=20,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # rpi4 out of mem
        # self.resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)
        self.db_f = db_f 
        self.image_path = images_path
        self.store = FaceStore(self.db_f)
        self.face_size = (160, 160)
        self.embeddings = []
        self.faces = []
        self.matches = []
        self.count = 0

    def __extract_faces(self, image_inf):
        image,_,image_path = image_inf
        boxes, probs, points = self.mtcnn.detect(image, landmarks=True)
        print("detected:", len(boxes))
        def f_path():
            return "/var/tmp/{}/F{}.png".format(
                self.db_f.replace("/", "_"),
                self.get_uid())   
        def f_norm(image_tensor):
            processed_tensor = (image_tensor - 127.5) / 128.0
            return processed_tensor
        face_paths = [f_path() for i in probs]
        fts = [f_norm(extract_face(image, box, save_path=face_path))
                   for (box, face_path) in list(zip(boxes, face_paths))]
        print("extracted: ", len(fts), fts[0].shape)
        image_paths= [image_path  for i in probs ]
        return (image_paths, face_paths, vfint(boxes), probs, fts)
    
    #torch.Size([512])
    def dist(self, e1, e2):
        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # d = cos(e1, e2).item()
        # d = LA.norm(e1-e2).item()
        d = (e1 - e2).norm().item()
        return d

    def compare_matches(self, faces, es):
        match_list = []
        for i, ei in enumerate(es):
            for j, ej in enumerate(self.embeddings):
                s = faces[i]
                t = self.faces[j]
                score = self.dist(ei, ej)
                m = (s[1], t[1], score)
                match_list.append(m)
                print("match = ", m)
        print("matched = [%d]" % len(match_list))            
        return match_list

    def cnnid_faces(self, fts):
        print("match_faces started ...")
        afts = torch.stack(fts).to(self.device)
        embedds = self.resnet(afts).detach().cpu()
        print("cnnid = ", embedds.shape)
        return embedds

    def extract_faces(self, image_path):
        print("started [%s]" % image_path)
        if len(self.store.find_image_path(image_path)) == 0:
            image_info = get_image(image_path)
            faces = self.__extract_faces(image_info)
            return faces
        else:
            print("duplicate")
            return () # FIXME

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
        return (len(self.faces),len(self.embeddings),len(self.matches),self.db_f)

def main():
    if len(sys.argv) < 3:
        raise "expect 2 params db_f, images_path"
    db_f, images_path = sys.argv[1], sys.argv[2]
    fsup = FaceUsurper(db_f, images_path)
    for image_path in get_image_files(images_path):
        i,f,b,c,t = fsup.extract_faces(image_path)
        faces = list(zip(i,f,b,c,t))
        fsup.update(faces=faces)

        ems = fsup.cnnid_faces(t)
        if len(fsup.embeddings) < 1:
            fsup.update(embeddings=ems)
            continue
      
        cms = fsup.compare_matches(faces, ems)
        fsup.update(matches=cms)
        fsup.update(embeddings=ems)
    
    print_nice(fsup.stats())

main()
