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


class Transformation:
    """

    """
    def __init__(self, config):
        """
        :param config: configuration containing the following parameters:
        top: y position for crop
        left: x position for crop
        image_height: height of original image
        image_width: width of original image
        crop_height: height of cropped image
        crop_width: width of cropped image
        out_channels: number of channels, 3 == rgb, 1 == gray
        mean: mean values for normalization
        std: standard deviation values for normalization
        """
        self.top = config['top']
        self.left = config['left']
        self.crop_height = config['crop_height']
        self.crop_width = config['crop_width']
        # self.out_channels = config['out_channels']
        # self.mean = config['mean']
        # self.std = config['std']
        self.transformation = t.Compose([
            Resize(height=110, width=84, interpolation=Image.BILINEAR),
            Crop(top=self.top, left=self.left, height=self.crop_height, width=self.crop_width),
            # t.ToTensor(),
            # t.Normalize(mean=self.mean, std=self.std),
            # t.ToPILImage(),
            t.Grayscale(num_output_channels=1),
            t.ToTensor()
        ])

    def transform(self, image):
        """
        Transforms image to reduce computation power. Constructs transformation pipeline and processes given image
        :param image: image to process
        :return: tensor for neural network input
        """
        image = Image.fromarray(image)
        return self.transformation(image).unsqueeze(dim=0)


class TransformationGridResize:
    def __init__(self):
        self.transformation = t.Compose([
            t.Resize((84, 84), interpolation=Image.NEAREST),
            t.Grayscale(num_output_channels=1),
            t.ToTensor()
        ])
    
    def transform(self, image):
        image = Image.fromarray(image)
        return self.transformation(image).unsqueeze(dim=0)


class TransformationGrid:
    def __init__(self):
        self.transformation = t.Compose([
            t.Grayscale(num_output_channels=1),
            t.ToTensor()
        ])

    def transform(self, image):
        image = Image.fromarray(image)
        return self.transformation(image).unsqueeze(dim=0)
