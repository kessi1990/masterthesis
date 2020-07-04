import torch
import torchvision.transforms as t
import torchvision.transforms.functional as tf
from PIL import Image


class Crop:
    """
    Crop transformation, crops image at specific location
    """
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, image):
        return tf.crop(image, self.top, self.left, self.height, self.width)


class Resize:
    """
    Resize transformation, resizes image to desired size
    Interpolation method can be set, default is BICUBIC interpolation
    """
    def __init__(self, height, width, interpolation=Image.BICUBIC):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, image):
        return tf.resize(image, (self.height, self.width), self.interpolation)


def transform_img(image):
    """
    Transforms image to reduce computation power. Constructs transformation pipeline and processes given image
    :param image: image to process
    :return: tensor for neural network input
    """
    # TODO load environment specific crop and resize parameters from config
    image = Image.fromarray(image)
    transformation = t.Compose([
        Resize(height=int(image.height / 2), width=int(image.width / 2), interpolation=Image.BICUBIC),
        Crop(top=16, left=4, height=84, width=72),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        t.ToPILImage(),
        t.Grayscale(num_output_channels=1),
        t.ToTensor()
    ])
    return transformation(image)
