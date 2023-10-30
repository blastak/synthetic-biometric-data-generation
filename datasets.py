import os

import torch
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path

        self.real_images = [f for f in os.listdir(image_folder_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]

        image_width = 64
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_width),
            transforms.CenterCrop(image_width),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_folder_path, self.real_images[index])
        img = Image.open(img_path).convert('RGB')
        tensor = self.tf(img)
        noise = torch.randn(512)
        sample = {'latent_vector': noise, 'real_image': tensor}

        return sample

    def __get_transform(self, augmentation):
        pass
