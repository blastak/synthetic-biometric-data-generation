from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from bio_modals.iris import Iris

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def filter_path(path_list, modality, DB_name):
    if modality == 'iris' and DB_name == 'IITD':
        i = 0
        while i < len(path_list):
            if 'Normalized_Images' in path_list[i].as_posix():
                path_list.pop(i)
            else:
                i += 1
    return path_list


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path):
        self.image_path_list = list(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        self.image_path_list = filter_path(self.image_path_list, 'iris', 'IITD')

        image_width = 64
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_width, image_width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        real_image = self.tf(img)
        noise = torch.randn(512)
        sample = {'latent_vector': noise, 'real_image': real_image}
        return sample


class EnhancementDataset(torch.utils.data.Dataset):
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    shrink_width = 64
    tf_condi = transforms.Compose([transforms.Resize(shrink_width, antialias=True), transforms.Resize(image_width, antialias=True)])

    def __init__(self, image_folder_path):
        self.image_path_list = list(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        self.image_path_list = filter_path(self.image_path_list, 'iris', 'IITD')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        real_image = self.tf_real(img)
        condi_image = self.tf_condi(real_image)
        sample = {'condition_image': condi_image, 'real_image': real_image}
        return sample


class IDPreserveDataset(torch.utils.data.Dataset):
    image_width = 320
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    tf_condi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize(0.5, 0.5)
    ])

    def __init__(self, image_path_list):
        self.image_path_list = list(p.resolve() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        w, h = img.size
        img_real = img.crop((w // 2, 0, w, h))
        real_image = self.tf_real(img_real)

        img_condi = img.crop((0, 0, w // 2, h))
        r, g, b = img_condi.split()
        img_condi = np.stack([b, r], axis=2)  # "r" is same as "g"
        condi_image = self.tf_condi(img_condi)

        sample = {'condition_image': condi_image, 'real_image': real_image}
        return sample
