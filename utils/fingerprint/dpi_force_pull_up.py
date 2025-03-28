# No.01
"""
DPI가 0으로 읽히는 BMP들을 불러와서 500으로 만들어서 다시 저장하는 코드.
DPI 500은 NFIQ2.exe를 돌리는데에 필요하다
"""

from PIL import Image
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\only_normal_30000", help='input image folder')
    ap.add_argument('--out_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\only_normal_30000_dpi500", help='output image folder')
    ap.add_argument('--dpi', type=int, default=500, help='dpi value for converting')
    opt = ap.parse_args()

    IN_DIR = opt.in_dir
    OUT_DIR = opt.out_dir
    DPI = opt.dpi

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']

    # 이미지 리스트 불러오기
    img_paths = [p.absolute() for p in Path(IN_DIR).glob('**/*') if p.suffix.lower() in IMG_EXTENSIONS]
    assert len(img_paths) != 0, 'empty list'

    # 폴더 없으면 만들기
    os.makedirs(OUT_DIR, exist_ok=True)

    for p in img_paths:
        img = Image.open(p).convert('L')
        out_path = Path(OUT_DIR) / p.name
        img.save(out_path, dpi=(DPI,DPI))
