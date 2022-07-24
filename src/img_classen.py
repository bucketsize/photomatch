# import the necessary packages
import torch
# specify image dimension
IMAGE_SIZE = 224
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# specify path to the ImageNet labels
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"

# import the necessary packages
from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}
class ImageClassen:
    def __init__(self):
        self.workers = 0 if os.name == 'nt' else 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        self.model = MODELS["vgg16"].eval().to(self.device)
        self.imagenet_tags = dict(enumerate(open(IN_LABELS)))

    def preprocess_image(self, image):
        # swap the color channels from BGR to RGB, resize it, and scale
        # the pixel values to [0, 1] range
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        image = image.astype("float32") / 255.0
        # subtract ImageNet mean, divide by ImageNet standard deviation,
        # set "channels first" ordering, and add a batch dimension
        image -= config.MEAN
        image /= config.STD
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        # return the preprocessed image
        return image

    def load_image(self, image_path):
        print("loading image %" % image_path)
        image = cv2.imread(image_path)
        orig = image.copy()
        image = preprocess_image(image)
        # convert the preprocessed image to a torch tensor and flash it to
        # the current device
        image = torch.from_numpy(image)
        image = image.to(self.device)

    def classify(self, image_path):
        logits = self.model(load_image(image_path))
        probabilities = torch.nn.Softmax(dim=-1)(logits)
        sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
        # loop over the predictions and display the rank-5 predictions and
        # corresponding probabilities to our terminal
        for (i, idx) in enumerate(sortedProba[0, :5]):
            print("{}. {}: {:.2f}%".format
                (i, imagenetLabels[idx.item()].strip(),
                probabilities[0, idx.item()] * 100))
