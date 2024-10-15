# No.06
"""
다음 표를 만족하는 리스트를 추출하기 위한 코드
1. 동공 (단위: 원둘레)
동공작음(S)      cir<237
동공보통(M) 237<=cir<287
동공큼 (L)  287<=cir
2. 홍채 보이는 정도 (단위: 퍼센트)
가려짐(O) vis<80 %
보통(N)   80<=vis %
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image
from circle_fit import taubinSVD

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iriscode_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iriscode', help='iris-code data folder')
    ap.add_argument('--npz_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iris_pupil', help='npz data folder')
    ap.add_argument('--out_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_statistics', help='out path to save file')
    opt = ap.parse_args()

    IRISCODE_ROOT = opt.iriscode_root
    NPZ_ROOT = opt.npz_root
    OUT_DIR = opt.out_dir

    pis = Path(IRISCODE_ROOT)
    pn = Path(NPZ_ROOT)
    po = Path(OUT_DIR)

    segmap_paths = list(pis.glob('**/*.tiff'))
    assert len(segmap_paths) != 0

    iris_vis_category = ['occlu', 'normal']
    pupil_size_category = ['smalls', 'mediums', 'larges']
    for c in product(iris_vis_category, pupil_size_category):
        locals()['_'.join(c)] = []  # 빈 리스트 선언

    for p, segmap_path in enumerate(segmap_paths):
        segmap = Image.open(segmap_path).convert('L')
        segmap_np = np.uint8(segmap)
        white_pixel = np.count_nonzero(segmap_np)
        vis = white_pixel / np.size(segmap_np)  # iris visibility (non-occluded ratio)

        npz_path = pn / segmap_path.relative_to(pis).with_suffix('.npz')
        if not npz_path.exists():
            print('no npz', npz_path)
            continue
        inners = np.load(npz_path.as_posix())['inners']
        _, _, ri, _ = taubinSVD(inners)  # circle fitting
        cir = 2 * np.pi * ri  # pupil_circumference

        # category determination
        vis_cat = 0 if vis < 0.8 else 1
        size_cat = 0 if cir < 237 else 1 if cir < 287 else 2

        var_name = iris_vis_category[vis_cat] + '_' + pupil_size_category[size_cat]
        locals()[var_name].append(segmap_path.relative_to(pis).with_suffix('.bmp').as_posix())

    # 저장
    for c in product(iris_vis_category, pupil_size_category):
        print('_'.join(c), len(locals()['_'.join(c)]))  # 갯수 분포 확인
        var_name = '_'.join(c)
        txt_path = po / Path('_'.join(c) + '.txt')
        # with open(txt_path, 'w', encoding='utf-8') as f:
        #     f.writelines(line + '\n' for line in locals()[var_name])  # Uncomment to use
