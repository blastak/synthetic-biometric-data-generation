# No.07
"""
원본(random choose)의 segmap과 생성(consecutive choose)의 segmap 간의 coverage 계산
GAN condition image 저장
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iriscode_root', type=str, default=r'D:\Dataset\02_Iris\90_Generated\20231226_DDPM\docker01', help='iris-code data folder')
    ap.add_argument('--iriscode_root', type=str, default=r'D:\Dataset\02_Iris\90_Generated\20231226_DDPM\docker01', help='generated data folder')
    ap.add_argument('--out_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_reconstruct', help='save folder')
    opt = ap.parse_args()

    IRISCODE_ROOT = opt.iriscode_root
    OUT_DIR = opt.out_dir
    assert IRISCODE_ROOT != OUT_DIR, '입력과 출력 경로가 같을 수 없음'

    size_recon = 256
    piece_w = 18
    piece_h = 66
    pic = Path(IRISCODE_ROOT)
    po = Path(OUT_DIR)

    img_paths = list(p.absolute() for p in pic.glob('**/*.bmp'))
    segmap_paths = list(p.absolute() for p in pic.glob('**/*.tiff'))
    assert len(img_paths) != 0
    assert len(img_paths) == len(segmap_paths)

    # 미완성. ABM 만드는 구문 추가해야함
