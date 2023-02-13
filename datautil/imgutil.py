import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


def get_transform(datasetName='idrid'):
    if datasetName == 'idrid':
        n = 4
        tra_transform = transforms.Compose([
            transforms.Resize((256 * n, 256 * n)),
            transforms.RandomCrop(224 * n),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            rotate(),
            transforms.RandomHorizontalFlip(),
            Cutout(n_holes=2, length=100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((256 * n, 256 * n)),
            transforms.CenterCrop(224 * n),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif datasetName == 'messidor':
        n = 2
        tra_transform = transforms.Compose([
            transforms.Resize((256 * n, 256 * n)),
            transforms.RandomCrop(224 * n),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            Cutout(n_holes=2, length=50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((256 * n, 256 * n)),
            transforms.CenterCrop(224 * n),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return tra_transform, val_transform


class Cutout(object):
    def __init__(self, n_holes=2, length=30):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        img = np.array(img)
        mask_val = img.mean()
        for _ in range(self.n_holes):
            top = np.random.randint(0 - self.length//2, img.shape[0] - self.length)
            left = np.random.randint(0 - self.length//2, img.shape[1] - self.length)
            bottom = top + self.length
            right = left + self.length
            top = 0 if top < 0 else top
            left = 0 if left < 0 else top
            img[top:bottom, left:right, :] = mask_val
        img = Image.fromarray(img)
        return img


class rotate():
    def __call__(self, img):
        angle = random.sample([0, 90, 180, 270, 360], 1)[0]
        return TF.rotate(img, angle)