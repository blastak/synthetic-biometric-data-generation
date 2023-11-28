"""
util02 로 저장한 npz와 이미지로 iris code(576x66)를 만들어 낸다
이때 레이블도 같이 변환하여 저장해둔다
"""

import os
from pathlib import Path

import cv2
import numpy as np

npz_path1 = r'D:\Dataset\02_Iris\IITD\IITD_Database'
npz_path2 = r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval'

npz_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\IITD\detected').glob('**/*') if p.suffix == '.npz')
npz_path_list.extend(sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\detected').glob('**/*') if p.suffix == '.npz'))

out_path = r'D:\Dataset\02_Iris\code576x66_IITD_CASIA'
os.makedirs(out_path, exist_ok=True)

target_height = 66
target_width_step = 18
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
    else:
        raise AssertionError
    hh, ww = img_np.shape[:2]

    loaded = np.load(npz_path_list[idx].as_posix())
    inners = loaded['inners']
    outers = loaded['outers']

    inners1 = np.roll(inners[::-1], 1, axis=0)  # 역순으로 바꾸기
    outers1 = np.roll(outers[::-1], 1, axis=0)
    target_img = np.empty([target_height, 0])
    target_lbl = np.empty([target_height, 0])
    for i in range(len(inners1)):
        p_tl = inners1[i]
        p_bl = outers1[i]
        try:
            p_tr = inners1[i + 1]
            p_br = outers1[i + 1]
        except:
            p_tr = inners1[0]
            p_br = outers1[0]
        pts = np.float32([p_tl, p_tr, p_br, p_bl])  # 반시계
        pts2 = np.float32([[0, 0], [target_width_step, 0], [target_width_step, target_height], [0, target_height]])
        H = cv2.getPerspectiveTransform(pts, pts2)
        piece = cv2.warpPerspective(img_np, H, [target_width_step, target_height])
        target_img = np.hstack([target_img, piece])
        piece = cv2.warpPerspective(lbl_np, H, [target_width_step, target_height])
        target_lbl = np.hstack([target_lbl, piece])
    target_img = target_img.astype(dtype=np.uint8)
    target_lbl = target_lbl.astype(dtype=np.uint8)
    _, target_lbl = cv2.threshold(target_lbl, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(out_path, n + ie),target_img)
    cv2.imwrite(os.path.join(out_path, n + '.tiff'),target_lbl)
