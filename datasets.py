import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    # '.tif', '.TIF', '.tiff', '.TIFF',
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
        self.image_path_list = sorted(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        if 'iris' in image_folder_path.lower():
            self.image_path_list = filter_path(self.image_path_list, 'iris', 'IITD')

        image_width = 64
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_width, image_width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        real_image = self.tf(img)
        noise = torch.randn(512, 1, 1)
        sample = {'latent_vector': noise, 'real_image': real_image}
        return sample


class ThumbnailIriscodeDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path):
        self.image_path_list = sorted(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)

        self.image_width = 64
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.image_width, self.image_width), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('L')
        real_image = self.tf(img)
        noise = torch.randn(512, 1, 1)
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
        self.image_path_list = sorted(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        if 'iris' in image_folder_path.lower():
            self.image_path_list = filter_path(self.image_path_list, 'iris', 'IITD')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        real_image = self.tf_real(img)
        condi_image = self.tf_condi(real_image)
        if random.random() > 0.5:
            tf = transforms.RandomHorizontalFlip(p=1.)
            real_image = tf(real_image)
            condi_image = tf(condi_image)
        sample = {'condition_image': condi_image, 'real_image': real_image}
        return sample


class EnhancementIriscodeDataset(torch.utils.data.Dataset):
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    patch_size = 64

    def __init__(self, image_folder_path):
        self.image_path_list = sorted(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('L')
        padded_img = Image.new(mode='L', size=(img.width, img.width))
        x, y = 0, padded_img.height // 2 - img.height // 2
        padded_img.paste(img, (x,y))
        real_image = self.tf_real(padded_img)

        _, ly, lx = os.path.splitext(os.path.basename(self.image_path_list[index]))[0].split('_')
        ly = int(ly)
        lx = int(lx)
        crop = img.crop((lx, ly, lx + self.patch_size, ly + self.patch_size))
        padded_crop = Image.new(mode='L', size=(img.width, img.width))
        padded_crop.paste(crop, (x + lx, y + ly))
        condi_image = self.tf_real(padded_crop)

        sample = {'condition_image': condi_image, 'real_image': real_image}
        return sample


class IDPreserveDataset(torch.utils.data.Dataset):
    # image_width = 320
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    condition_channels = 2
    tf_condi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * condition_channels, [0.5] * condition_channels)
    ])

    def __init__(self, image_path_list):
        self.image_path_list = sorted(p.resolve() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')

        # right-side is real image
        w, h = img.size
        img_real = img.crop((w // 2, 0, w, h))
        real_image = self.tf_real(img_real)

        # left-side is condition image
        img_condi = img.crop((0, 0, w // 2, h))
        r, g, b = img_condi.split()
        img_condi = np.stack([b, r], axis=2)  # "r" is same as "g"
        condi_image = self.tf_condi(img_condi)

        sample = {'condition_image': condi_image, 'real_image': real_image}
        return sample


class IDPreserveTwoDataset(torch.utils.data.Dataset):
    # image_width = 320
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    condition_channels = 2
    tf_condi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * condition_channels, [0.5] * condition_channels)
    ])

    def __init__(self, image_path_list1, image_path_list2):
        self.image_path_list1 = sorted(p.resolve() for p in Path(image_path_list1).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        self.image_path_list2 = sorted(p.resolve() for p in Path(image_path_list2).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list1)

    def __getitem__(self, index):
        img1 = Image.open(self.image_path_list1[index]).convert('RGB')
        condi_image = self.tf_real(img1)

        img2 = Image.open(self.image_path_list2[index]).convert('RGB')
        target_image = self.tf_real(img2)

        sample = {'condition_image': condi_image, 'real_image': target_image}
        return sample


class IDPreservePairMaskDataset(torch.utils.data.Dataset):
    # image_width = 320
    image_width = 256
    tf_real = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    condition_channels = 2
    tf_condi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5] * condition_channels, [0.5] * condition_channels)
    ])
    tf_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_width, image_width), antialias=True),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(self, image_path_list):
        self.image_path_list = sorted(p.resolve() for p in Path(image_path_list).glob('**/*') if p.suffix in IMG_EXTENSIONS)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index])

        w, h = img.size
        w2 = int(w / 3)
        pair_left = img.crop((0, 0, w2, h))
        pair_right = img.crop((w2, 0, w2 * 2, h)).convert('L')
        mask = img.crop((w2 * 2, 0, w, h)).convert('L')

        t_real = self.tf_real(pair_right)
        t_mask = self.tf_mask(mask)

        r, g, b = pair_left.split()
        left = np.stack([r, b], axis=2)
        t_condi = self.tf_condi(left)
        t_condi = torch.cat((t_condi, t_mask), dim=0)

        sample = {'condition_image': t_condi, 'real_image': t_real}
        return sample
