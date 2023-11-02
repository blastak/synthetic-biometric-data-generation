from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

from bio_modals.iris import Iris

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

# def filter_path(path_list, modality, DB_name):
#     if modality == 'iris' and DB_name == 'IITD':
#         i = 0
#         while i < len(path_list):
#             if 'Normalized_Images' in path_list[i].as_posix():
#                 path_list.pop(i)
#             else:
#                 i += 1
#     return path_list


class ThumbnailDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path):
        self.image_path_list = list(p.resolve() for p in Path(image_folder_path).glob('**/*') if p.suffix in IMG_EXTENSIONS)
        # self.image_path_list = filter_path(self.image_path_list, modality, 'IITD')

        image_width = 64
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_width),
            transforms.CenterCrop(image_width),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img = Image.open(self.image_path_list[index]).convert('RGB')
        tensor = self.tf(img)
        noise = torch.randn(512)
        sample = {'latent_vector': noise, 'real_image': tensor}
        return sample

