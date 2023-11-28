"""
util03 으로 저장한 iris code(576x66)에서 64x64를 만들어 낸다
"""

import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\code576x66_IITD_CASIA').glob('**/*') if p.suffix in IMG_EXTENSIONS)
lbl_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\code576x66_IITD_CASIA').glob('**/*') if p.suffix == '.tiff')

out_path = r'D:\Dataset\02_Iris\for_thumbnail_train'
os.makedirs(out_path, exist_ok=True)

target_height = 66
target_width_step = 18
for idx in range(len(image_path_list)):
    print(idx + 1, '/', len(image_path_list))
    d, f = os.path.split(image_path_list[idx])
    n, e = os.path.splitext(f)

    target_img = cv2.imread(image_path_list[idx].as_posix(), cv2.IMREAD_GRAYSCALE)
    target_lbl = cv2.imread(lbl_path_list[idx].as_posix(), cv2.IMREAD_GRAYSCALE)

    ## 576x66 의 중간에서부터 64x64 잘라보기
    for jjjjj in range(2):
        patch_size = 64
        offset_y = target_img.shape[0] // 2 - patch_size // 2
        offset_x = target_img.shape[1] // 2 - patch_size // 2
        if jjjjj != 0:
            offset_x = 0
        score = []
        for i in [0, -1, 1]:
            y = offset_y + i
            for j in range(33):
                x = offset_x + j
                patch_lbl = target_lbl[np.ix_(range(y,y + patch_size), range(x,x + patch_size))]
                score.append([(patch_lbl == 255).sum(), i, j])
                x = offset_x - j
                patch_lbl = target_lbl[np.ix_(range(y,y + patch_size), range(x,x + patch_size))]
                score.append([(patch_lbl == 255).sum(), i, -j])
        score.sort(key=lambda x: (-x[0], abs(x[1]), abs(x[2])))
        lx, ly = offset_x + score[0][2], offset_y + score[0][1]

        patch_img = target_img[np.ix_(range(ly, ly + patch_size), range(lx, lx + patch_size))]
        fname = '%s_%d_%d%s' % (n, ly, lx, e)
        cv2.imwrite(os.path.join(out_path, fname), patch_img)
        print(fname, 'saved')
