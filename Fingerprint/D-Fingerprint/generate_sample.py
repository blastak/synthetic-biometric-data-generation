import os
import argparse
import cv2
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
#from scipy.misc import imread, imsave, imresize
from train import DFingerprintGenerator


def d_fp_generation(path):
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(a)
    img_A = np.stack((b, g), 2)  # "g" is same as "r"
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((320, 320), antialias=True)])
    real_A = tf(img_A)
    real_A = torch.unsqueeze(real_A, dim=0).to(device)

    DFinger_batch = Gen_DFinger(real_A)  # ?

    img_DFinger = DFinger_batch.detach().cpu()
    montage_DFinger = make_grid(img_DFinger, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
    montage_DFinger = cv2.normalize(montage_DFinger, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                    dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imshow('DFinger', montage_DFinger)
    save_image_1_bmp = cv2.cvtColor(montage_DFinger, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('test.bmp', save_image_1_bmp)
    cv2.waitKey(0)


###########

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_DFingerprint', type=str, default='../../../model/ckpt_epoch400.pth')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_path', type=str, default= r"C:\Users\CVlab\Desktop\Neurotec_Biometric_12_4_SDK_2023-05-17\Neurotec_Biometric_12_4_SDK\Bin\Digent_test_sample\Minutiae\3_00000020_DIG00_L1_01_N.png")
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
    d_fp_generation(path)