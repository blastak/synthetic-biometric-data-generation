import argparse
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms

from train_thumbnailGAN import ThumbnailGenerator
from train_ridgepatternGAN import RidgePatternGenerator

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_thumbnail', type=str, default='weights/thumbnail_gan_001/ckpt_epoch190.pth')
    parser.add_argument('--ckpt_path_ridgepattern', type=str, default='weights/ridgepattern_gan_001/ckpt_epoch200.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    bs = args.batch_size

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    ckpt_thumbnail = torch.load(args.ckpt_path_thumbnail,map_location=device)
    Gen_thumbnail = ThumbnailGenerator()
    Gen_thumbnail.load_state_dict(ckpt_thumbnail['modelG_state_dict'])
    Gen_thumbnail.to(device)
    Gen_thumbnail.eval()

    ckpt_ridgepattern = torch.load(args.ckpt_path_ridgepattern,map_location=device)
    Gen_ridgepattern = RidgePatternGenerator()
    Gen_ridgepattern.load_state_dict(ckpt_ridgepattern['modelG_state_dict'])
    Gen_ridgepattern.to(device)
    Gen_ridgepattern.eval()

    noise = torch.randn(bs, 512)
    thumbnail_batch = Gen_thumbnail(noise)

    img_thumbnail = thumbnail_batch.detach().cpu()
    montage_thumbnail = make_grid(img_thumbnail, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
    montage_thumbnail = cv2.normalize(montage_thumbnail, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imshow('thumbnail', montage_thumbnail)
    cv2.waitKey(1)

    image_size = 256
    tf = transforms.Resize(image_size,antialias=True)
    real_A = tf(thumbnail_batch)
    img_fake_B = Gen_ridgepattern(real_A).detach().cpu()
    montage_fake_B = make_grid(img_fake_B, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
    montage_fake_B = cv2.normalize(montage_fake_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imshow('ridge pattern', montage_fake_B)
    cv2.waitKey(0)
