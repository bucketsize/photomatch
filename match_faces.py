import time
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

# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# from scipy.spatial.distance import cosine

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

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
def get_faces(image_inf, required_size=(224, 224)):
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

store = FaceStore()
def process(image_path, data=[]):
    if len(store.find_image_path(image_path)) == 0:
        pt1=time.perf_counter()
        image_info = get_image(image_path)
        faces = get_faces(image_info)
        # print_nice(faces)
        images = get_images(image_info, faces)
        pt2=time.perf_counter()
        print("processed [%s] in [%d]s" % (image_path, pt2-pt1))
        #data.append({"image_path": image_path, "rois": faces})
        store.save_face(image_path, faces)
        # show_overlay(image_info, faces)
    else:
        print("duplicate [%s]" % (image_path))

def main():
    data = []
    for i in get_image_files("test"):
         process(i, data=data)
    print_nice(data)

main()
