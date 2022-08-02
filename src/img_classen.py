# https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/#download-the-code

import sys
import os
import numpy as np
import cv2
import torch
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import Inception_V3_Weights
from torchvision.models import DenseNet121_Weights 
from face_utils import get_image_files
from img_store import ImgStore

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"
MODELS = {
	# "vgg16": models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
	# "vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1),
	# "densenet": models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
	# "resnet": models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
}
class ImageClassen:
    def __init__(self, pt_model, db_f):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('ii running on device: {}'.format(self.device))
        self.model = pt_model.eval().to(self.device)
        self.imagenet_tags = dict(enumerate(open(IN_LABELS)))
        self.store = ImgStore(db_f)

    def preprocess_image(self, image):
        # swap the color channels from BGR to RGB, resize it, and scale
        # the pixel values to [0, 1] range
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype("float32") / 255.0
        # subtract ImageNet mean, divide by ImageNet standard deviation,
        # set "channels first" ordering, and add a batch dimension
        image -= MEAN
        image /= STD
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        # return the preprocessed image
        return image

    def load_image(self, image_path):
        print("ii loading image %s" % image_path)
        image = cv2.imread(image_path)
        orig = image.copy()
        image = self.preprocess_image(image)
        image = torch.from_numpy(image)
        image = image.to(self.device)
        return image


    def classify(self, image_path):
        logits = self.model(self.load_image(image_path))
        classs = torch.nn.Softmax(dim=-1)(logits)
        sclass = torch.argsort(classs, dim=-1, descending=True)
        for (i, idx) in enumerate(sclass[0, :2]):
            print("ii {}. {}: {:.2f}%".format
                (i, self.imagenet_tags[idx.item()].strip(),
                classs[0, idx.item()] * 100))
            if i == 0:
                self.store.save_image(
                     ( image_path
                     , idx.item()
                     , self.imagenet_tags[idx.item()].strip())
                )

def main():
    if len(sys.argv) < 4:
        raise "expect model image_path db_f"
    pm = MODELS[sys.argv[1]]
    cn = ImageClassen(pm, sys.argv[3])
    for image_path in get_image_files(sys.argv[2]):
        if len(cn.store.find_image_path(image_path)) > 0:
            print("ii dup "+image_path)
        else:
            try:
                cn.classify(image_path)
            except:
                print("ii skip on error: %s" % image_path)
                pass

main()
