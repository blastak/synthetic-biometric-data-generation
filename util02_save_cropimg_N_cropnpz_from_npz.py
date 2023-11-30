"""
이 파일은 Path 내의 모든 npz 파일로 이미지와 레이블을 자르고 detected 폴더에 저장한다
이때 npz도 offset을 설정하여 다시 저장한다.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# npz_path = r'D:\Dataset\02_Iris\IITD\IITD_Database'
# npz_path = r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval'
npz_path = r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\train\images'

npz_path_list = sorted(p.resolve() for p in Path(npz_path).glob('**/*') if p.suffix == '.npz')

out_path = os.path.join(os.path.split(npz_path)[0], 'detected')
os.makedirs(out_path, exist_ok=True)

for idx in range(len(npz_path_list)):
    print(idx + 1, '/', len(npz_path_list))

    d, f = os.path.split(npz_path_list[idx])
    n, e = os.path.splitext(f)

    for ie in ['.bmp', '.jpg', '.png']:
        if Path(d, n + ie).exists():
            img_np = cv2.imread(os.path.join(d, n + ie), cv2.IMREAD_GRAYSCALE)
            break
    if Path(d, n + '.tiff').exists():
        lbl_np = cv2.imread(os.path.join(d, n + '.tiff'), cv2.IMREAD_GRAYSCALE)
    hh, ww = img_np.shape[:2]

    loaded = np.load(npz_path_list[idx].as_posix())
    inners = loaded['inners']
    outers = loaded['outers']
    cx, cy = map(int, sum(inners) / len(inners))


    def cropp(img):
        hh_target = ww // 4 * 3
        top = int(cy - hh_target / 2.)  ## top
        bottom = top + hh_target  ## bottom
        if top < 0:
            bottom -= top
            top = 0
        elif bottom > hh:
            top -= bottom - hh
            bottom = hh
        cr_img = img[top:bottom, 0:ww]
        return cr_img, top


    nf, e = npz_path_list[idx].relative_to(npz_path).as_posix().split('.')
    nf = nf.replace('/', '_')

    cr_img, top = cropp(img_np)
    cv2.imwrite(os.path.join(out_path, nf + ie), cr_img)
    if Path(d, n + '.tiff').exists():
        cr_lbl, _ = cropp(lbl_np)
        cv2.imwrite(os.path.join(out_path, nf + '.tiff'), cr_lbl)

    inners[:, 1] -= top
    outers[:, 1] -= top
    np.savez(os.path.join(out_path, nf + '.npz'), inners=inners, outers=outers)

    # img_disp1 = cv2.cvtColor(cr_img,cv2.COLOR_GRAY2BGR)
    # color_i = (0, 0, 255)
    # color_o = (0, 255, 0)
    # color_conn = (0, 255, 255)
    # for i in range(-1, len(inners) - 1):
    #     cv2.line(img_disp1, inners[i], inners[i + 1], color_i, 2)
    #     cv2.line(img_disp1, outers[i], outers[i + 1], color_o, 2)
    #     cv2.line(img_disp1, inners[i], outers[i], color_conn, 1)
    # cv2.imshow('img_disp1', img_disp1)
    # cv2.waitKey(0)
