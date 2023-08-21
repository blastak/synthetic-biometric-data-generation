import os
import argparse
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from train import DFingerprintGenerator
from PIL import Image
from PIL import Image

#load minutiae map
def convert_minutiae_to_real_A(path):
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(a)
    img_A = np.stack((b, g), 2)  # "g" is same as "r"
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((320, 320), antialias=True)])
    real_A = tf(img_A)
    real_A = torch.unsqueeze(real_A, dim=0).to(device)
    return real_A
###########
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_DFingerprint', type=str, default='../../../model/ckpt_epoch400.pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_path', type=str, default="../../../Dataset/3_00000004_DIG00_R3_04_B.png")
    args = parser.parse_args()
    bs = args.batch_size
    path = args.image_path
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    ckpt_DFingerprint = torch.load(args.ckpt_path_DFingerprint, map_location=device)
    Gen_DFinger = DFingerprintGenerator()
    Gen_DFinger.load_state_dict(ckpt_DFingerprint['modelG_state_dict'])
    Gen_DFinger.to(device)
    Gen_DFinger.eval()
    real_A = convert_minutiae_to_real_A(path)
    DFinger_batch = Gen_DFinger(real_A)
    img_DFinger = DFinger_batch.detach().cpu()
    montage_DFinger = make_grid(img_DFinger, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
    montage_DFinger = cv2.normalize(montage_DFinger, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F).astype(np.uint8)
    save_image_1_bmp = cv2.cvtColor(montage_DFinger, cv2.COLOR_BGR2GRAY)
    save_image_1_bmp = cv2.resize(save_image_1_bmp, dsize=(280, 320), interpolation=cv2.INTER_AREA)
    ## bmp DPI modify
    img = Image.fromarray(save_image_1_bmp)
    dpi = (500, 500)
    img.save('test.bmp', dpi=dpi)