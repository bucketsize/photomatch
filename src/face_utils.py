import glob
import ssl
import urllib.request
import pprint
from numpy import asarray
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

pp = pprint.PrettyPrinter(width=80, compact=True)
def print_nice(o):
    pp.pprint(o)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def join(sep, ls):
    s = ""
    for i in ls:
        if s == "":
            s = s+str(i)
        else:
            s = s+sep+str(i)
    return s        

def http_get_file(url, local_file_name):
    if os.path.exists(local_file_name):
        print("cachehit: %s", local_file_name)
        return 
    else:
        with urllib.request.urlopen(url, context=ctx) as resource:
            with open(local_file_name, 'wb') as f:
                f.write(resource.read())

def get_image_files(base_dir):
    return glob.iglob(base_dir+"/**/*.jpg", recursive=True)

def in_range(r, i):
    (x1, x2) = r
    return (x1<=i and i<=x2) 

def in_box(r, x, y):
    (x1, y1, x2, y2) = r
    return (x1<=x and x<=x2 and y1<=y and y<=y2)

def get_images(pil_image, rois, confidence=0.85, required_size=(224, 224)):
    images = []

    for roi in rois:
        x1, y1, x2, y2 = roi["box"] 
        # print("roi: ", x1, y1, x2, y2)
        c = roi["confidence"]
        w, h = iinfo.size
        if in_box((0,0,w,h), x1, y1) and in_box((0,0,w,h), x2, y2): 
            bb = pil_image[y1:y2, x1:x2]
            img = Image.fromarray(bb).resize(required_size)
            img_array = asarray(img)
            images.append(img_array)
        else:
            print("drop invalid roi [%s]" % (name), roi["box"])

    return images

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

