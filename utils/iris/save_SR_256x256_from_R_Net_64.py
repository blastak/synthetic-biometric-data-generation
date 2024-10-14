# No.05
"""
DCGAN이나 DDPM에서 생성된 64x64x1 x batch 크기의 이미지를 crop하고
256x256 으로 resize 후에 각각 EnhancementGAN 돌려서 선명하게 만들고
thresholding으로 segmap 까지 함께 저장해두기
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.R_Enhancement import EnhancementGAN


class MyUpScale:
    tf = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)])

    def __init__(self, ckpt_file_path):
        gpu_ids = [0]  # 어떤 gpu 사용할 지
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
        torch.set_default_device(self.device)

        self.EnhancementGAN = EnhancementGAN(1, 1, gpu_ids)
        self.load(ckpt_file_path)

    def load(self, ckpt_file_path):
        checkpoint = torch.load(ckpt_file_path, map_location=self.device)
        self.EnhancementGAN.net_G.load_state_dict(checkpoint['modelG_state_dict'])
        self.EnhancementGAN.net_G.to(self.device)
        self.EnhancementGAN.net_G.eval()

    def do_upscale(self, img):
        t_condition = self.tf(img).to(self.device)
        t_result = self.EnhancementGAN.net_G(t_condition.unsqueeze(1)).detach().cpu()  # 만약 배치로 연산하길 원한다면 unsqueeze를 바꾸어라
        np_img = np.uint8((t_result.numpy().squeeze() * 0.5 + 0.5) * 255)
        return np_img


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_root', type=str, default=r'D:\Dataset\02_Iris\ddpm_gen\epoch0219\iris_gen1', help='generated data folder')
    ap.add_argument('--aspect_ratio', type=int, default=4, help='aspect ratio as batch_size of generated image')
    ap.add_argument('--file_ext', type=str, default='.png', help='file extension')
    ap.add_argument('--ckpt_enhanceGAN', type=str, default=r'..\..\checkpoints\20231205B_Enhance_server3_tr001\ckpt_epoch000700.pth', help='generated data folder')
    ap.add_argument('--out_dir', type=str, default=r'D:\Dataset\02_Iris\90_Generated\20231226_DDPM\docker01', help='save folder')
    opt = ap.parse_args()

    GEN_ROOT = opt.gen_root
    ASPECT_RATIO = opt.aspect_ratio
    FILE_EXT = opt.file_ext
    OUT_DIR = opt.out_dir

    pg = Path(GEN_ROOT)
    po = Path(OUT_DIR)

    sr = MyUpScale(opt.ckpt_enhanceGAN)

    gen_paths = list(p.absolute() for p in pg.glob(f'**/*{FILE_EXT}'))
    assert len(gen_paths) != 0

    for i, gen_path in enumerate(gen_paths):
        img_batch = Image.open(gen_path).convert('L')  # 원본 이미지 grayscale로 로드
        w_step = img_batch.width // ASPECT_RATIO
        cnt = 1
        for b in range(ASPECT_RATIO):
            img_in = img_batch.crop((w_step * b, 0, w_step * (b + 1), img_batch.height))  # crop
            # Upscaling
            us_img = sr.do_upscale(img_in)  # upscale 64x64 --> 256x256
            cv2.circle(us_img, (us_img.shape[1] // 2, us_img.shape[0] // 2), 40 - 2, color=0, thickness=cv2.FILLED)  # 동공(pupil) 부분 검은색 채우기
            # getting segmap by thresholding
            _, us_segmap = cv2.threshold(us_img, 25, 255, cv2.THRESH_BINARY)  # 실험적으로 25로 결정
            # us_segmap의 fill gap (using morph_close3x3)
            up_segmap_closed = cv2.morphologyEx(us_segmap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

            # cv2.imshow('us_img',us_img)
            # cv2.imshow('up_segmap_closed',up_segmap_closed)
            # cv2.waitKey()

            # file save
            out_img_p = po / Path(gen_path.relative_to(pg).stem + '_%d.bmp' % cnt)
            cnt += 1
            # Image.fromarray(us_img).save(out_img_p)  # Uncomment to use
            out_segmap_p = out_img_p.with_suffix('.tiff')
            # Image.fromarray(up_segmap_closed).save(out_segmap_p)  # Uncomment to use
