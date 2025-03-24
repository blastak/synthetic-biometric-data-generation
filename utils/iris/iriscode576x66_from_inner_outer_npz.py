# No.03
"""
dataset_root와 유사한 폴더 구조로 저장되어 있는 segmap과 npz를 검색하고
npz의 32-sided polygon 정보를 토대로
576x66 크기의 iris_code 및 iris_code_segmap 정보를 저장한다.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database', help='original data folder')
    ap.add_argument('--segmap_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_segmap', help='segmap data folder')
    ap.add_argument('--npz_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iris_pupil', help='npz data folder')
    ap.add_argument('--out_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iriscode', help='save folder')
    opt = ap.parse_args()

    DATASET_ROOT = opt.dataset_root
    SEGMAP_ROOT = opt.segmap_root
    NPZ_ROOT = opt.npz_root
    OUT_DIR = opt.out_dir
    assert DATASET_ROOT != OUT_DIR, '입력과 출력 경로가 같을 수 없음'

    piece_w = 18
    piece_h = 66
    pd = Path(DATASET_ROOT)
    ps = Path(SEGMAP_ROOT)
    pn = Path(NPZ_ROOT)
    po = Path(OUT_DIR)

    img_paths = list(p.absolute() for p in pd.glob('**/*') if p.suffix.lower() in ['.bmp', '.jpg'])
    assert len(img_paths) != 0

    for p, img_p in enumerate(img_paths):
        # segmap, npz 존재 검색
        rp = img_p.relative_to(pd)
        segmap_p = list(ps.glob(rp.with_suffix('.tiff').as_posix()))
        npz_p = list(pn.glob(rp.with_suffix('.npz').as_posix()))
        if len(segmap_p) != 1 or len(npz_p) != 1:
            print('segmap 이나 npz가 없음(또는 중복)', p, rp)
            continue

        # 데이터 로드
        img = Image.open(img_p).convert('L')  # 원본 이미지 grayscale로 로드
        segmap = Image.open(segmap_p[0]).convert('L')  # segmap grayscale로 로드
        npz = np.load(npz_p[0].as_posix())

        # h=66, w=0 빈 이미지 만들어놓기
        iris_code_img = np.empty([piece_h, 0], dtype=np.uint8)
        iris_code_segmap = np.empty([piece_h, 0], dtype=np.uint8)
        ni = np.vstack([npz['inners'], npz['inners'][0]])  # 연장술
        no = np.vstack([npz['outers'], npz['outers'][0]])
        for i in range(32):
            p_tl, p_bl = ni[i + 1], no[i + 1]
            p_tr, p_br = ni[0], no[0]
            pts = np.float32([p_tl, p_tr, p_br, p_bl])  # CW
            pts2 = np.float32([[0, 0], [piece_w, 0], [piece_w, piece_h], [0, piece_h]])
            H = cv2.getPerspectiveTransform(pts, pts2)
            piece = cv2.warpPerspective(np.uint8(img), H, [piece_w, piece_h])
            iris_code_img = np.hstack([piece, iris_code_img])  # 오른쪽부터 채우기 (IITD의 normalized image와 좌표계를 유사하게 하기위해)
            piece = cv2.warpPerspective(np.uint8(segmap), H, [piece_w, piece_h])
            iris_code_segmap = np.hstack([piece, iris_code_segmap])
        _, iris_code_segmap = cv2.threshold(iris_code_segmap, 127, 255, cv2.THRESH_BINARY)  # 이진화를 하지 않으면 warping 중 발생한 gray value가 남아있음

        # file save
        out_img_p = po / rp.with_suffix('.bmp')
        out_img_p.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray(iris_code_img).save(out_img_p) # Uncomment to use
        out_segmap_p = out_img_p.with_suffix('.tiff')
        # Image.fromarray(iris_code_segmap).save(out_segmap_p) # Uncomment to use
