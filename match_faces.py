import time
import os
import glob
import ssl
import urllib.request
import pprint
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

# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# from scipy.spatial.distance import cosine

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def http_get_file(url, local_file_name):
    if os.path.exists(local_file_name):
        print("cachehit: %s", local_file_name)
        return 
    else:
        with urllib.request.urlopen(url, context=ctx) as resource:
            with open(local_file_name, 'wb') as f:
                f.write(resource.read())

pp = pprint.PrettyPrinter(width=41, compact=True)
def print_nice(o):
    pp.pprint(o)

# lazy, eager
def get_image(image_path):
    return (Image.open(image_path), plt.imread(image_path), image_path)

def show_image(image_path):
    plt.imshow(plt.imread(image_path))

# eager
def show_overlay(image, roi_boxes):
    plt.imshow(image)
    ax = plt.gca()

    for roi in roi_boxes:
        x1, y1, x2, y2 = roi["box"]
        roi_border = Rectangle((x1, y1), x2-x1, y2-y1,
                            fill=False, color='red')
        ax.add_patch(roi_border)
    plt.show()

# def collate_fn(x):
#     return x[0]

# dataset = datasets.ImageFolder('../data/test_images')
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

vfint = np.vectorize(int)

        # >>> from PIL import Image, ImageDraw
        # >>> from facenet_pytorch import MTCNN, extract_face
        # >>> mtcnn = MTCNN(keep_all=True)
        # >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        # >>> # Draw boxes and save faces
        # >>> img_draw = img.copy()
        # >>> draw = ImageDraw.Draw(img_draw)
        # >>> for i, (box, point) in enumerate(zip(boxes, points)):
        # ...     draw.rectangle(box.tolist(), width=5)
        # ...     for p in point:
        # ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        # ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        # >>> img_draw.save('annotated_faces.png')

# lazy
def get_faces_2(image_inf, required_size=(224, 224)):
    image,_,name = image_inf
    image_size,_ = required_size
    mtcnn = MTCNN(
        image_size=image_size, margin=40, min_face_size=20,
        select_largest=False,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
        device=device
    )
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    faces_list = []
    for i, (box, prob, point) in enumerate(zip(boxes, probs, points)):
        ftensor = extract_face(
            image, box,
            save_path='/tmp/photomatch/face_{}_{}.png'.format(name.replace("/", "."), i))
        faces_list.append({
            "box": vfint(box),
            "confidence": prob,
            "ftensor": ftensor
        })
    print("faces [%s] = [%d]" % (name, len(faces_list)))
    return faces_list

def get_faces_1(image_inf, required_size=(224, 224)):
    image,_,name = image_inf
    bounding_boxes, landmarks = detect_faces(image)
    print("faces [%s] = [%d]" % (name, len(bounding_boxes)))
    return list(map(lambda x: {"box": [int(y) for y in x[:-1]], "confidence":x[4]}, bounding_boxes)) 

def in_range(r, i):
    (x1, x2) = r
    return (x1<=i and i<=x2) 

def in_box(r, x, y):
    (x1, y1, x2, y2) = r
    return (x1<=x and x<=x2 and y1<=y and y<=y2)

# eager
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

def process(image_path, data=[]):
    print("started [%s]" % (image_path))
    pt1=time.perf_counter()
    
    image_info = get_image(image_path)
    faces = get_faces_2(image_info)
    # print_nice(faces)
    
    images = get_images(image_info, faces)

    pt2=time.perf_counter()

    data.append({"image_path": image_path, "rois": faces})
    print("processed [%s] in [%d]s" % (image_path, pt2-pt1))

    # show_overlay(image_info, faces)

def get_image_files(base_dir):
    return glob.iglob(base_dir+"/**/*.jpg", recursive=True)

def main():
    data = []
    for i in get_image_files("test"):
         process(i, data=data)
    print_nice(data)

main()
