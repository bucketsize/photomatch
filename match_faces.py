import time
import os
import glob
import ssl
import urllib.request
import pprint
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray
from PIL import Image
from torch_mtcnn import detect_faces

# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# from scipy.spatial.distance import cosine

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


# lazy
def get_faces(image_inf, required_size=(224, 224)):
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
    faces = get_faces(image_info)
    # print_nice(faces)
    
    images = get_images(image_info, faces)

    pt2=time.perf_counter()

    data.append({"image_path": image_path, "rois": faces})
    print("processed [%s] in [%d]s" % (image_path, pt2-pt1))

    # show_overlay(image_data, faces)

def get_image_files(base_dir):
    return glob.iglob(base_dir+"/**/*.jpg", recursive=True)

def main():
    data = []
    for i in get_image_files("test"):
         process(i, data=data)
    print_nice(data)

main()
