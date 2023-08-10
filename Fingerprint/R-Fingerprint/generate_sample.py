import argparse
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

from train_thumbnailGAN import Generator

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='weights/thumbnail_gan_001/ckpt_epoch100.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    mymodel_G = Generator()

    ckpt = torch.load(args.ckpt_path)
    mymodel_G.load_state_dict(ckpt['modelG_state_dict'])
    mymodel_G.to(device)

    mymodel_G.eval()

    noise = torch.randn(64, 512)
    img = mymodel_G(noise).detach().cpu()
    montage = make_grid(img, nrow=8, normalize=True).permute(1, 2, 0).numpy()
    norm_image = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    cv2.imshow('big', norm_image)
    cv2.waitKey(1)

    noise = torch.randn(1, 512)
    img = mymodel_G(noise).detach().cpu().numpy().squeeze()
    norm_image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    cv2.imshow('one', norm_image)
    cv2.waitKey(0)