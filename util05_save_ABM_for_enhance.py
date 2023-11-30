"""
util03 으로 저장한 iris code(576x66)로 ABM을 만든다
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\code576x66_IITD_CASIA').glob('**/*') if p.suffix in IMG_EXTENSIONS)
lbl_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\code576x66_IITD_CASIA').glob('**/*') if p.suffix == '.tiff')

mold_npz_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\validation\detected').glob('**/*') if p.suffix == '.npz')

out_path = r'D:\Dataset\02_Iris\for_enhance_ABM3\test'
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
    patch_size = 64
    offset_y = target_img.shape[0] // 2 - patch_size // 2
    offset_x = target_img.shape[1] // 2 - patch_size  # 생각보다 왼쪽이 좋을 듯(=원에서 왼쪽 아래)
    score = []
    for i in [0, -1, 1]:
        y = offset_y + i
        for j in range(51):
            x = offset_x + j
            patch_lbl = target_lbl[np.ix_(range(y, y + patch_size), range(x, x + patch_size))]
            score.append([(patch_lbl == 255).sum(), i, j])
            x = offset_x - j
            patch_lbl = target_lbl[np.ix_(range(y, y + patch_size), range(x, x + patch_size))]
            score.append([(patch_lbl == 255).sum(), i, -j])
    score.sort(key=lambda x: (-x[0], abs(x[1]), abs(x[2])))
    lx, ly = offset_x + score[0][2], offset_y + score[0][1]

    ### squaring

    # hh,ww=target_img.shape[:2]
    # A = np.zeros((ww,ww),dtype=np.uint8) # condition
    # B = np.zeros((ww,ww),dtype=np.uint8) # real
    # M = np.zeros((ww,ww),dtype=np.uint8) # mask
    #
    # # roll 을 랜덤으로 하는것도 고려해보자
    #
    # cp = ww//2-hh//2
    target_patch = np.zeros_like(target_img)
    target_patch[ly:ly + patch_size, lx:lx + patch_size] = target_img[ly:ly + patch_size, lx:lx + patch_size]
    # A[cp+ly:cp+ly+patch_size,lx:lx+patch_size] = target_img[ly:ly+patch_size,lx:lx+patch_size]
    # B[cp:cp+hh,:] = target_img
    # M[cp:cp+hh,:] = target_lbl
    # stacked = np.hstack([A,B,M])

    ### re-circle
    idx_mold = random.randint(0, len(mold_npz_list) - 1)
    loaded = np.load(mold_npz_list[idx_mold].as_posix())
    inners = loaded['inners']
    outers = loaded['outers']
    inners1 = np.roll(inners[::-1], 1, axis=0)  # 역순으로 바꾸기
    outers1 = np.roll(outers[::-1], 1, axis=0)
    cx, cy = map(int, sum(outers1) / len(outers1))
    ox, oy = 256 // 2 - cx, 256 // 2 - cy
    inners1[:, 0] += ox
    outers1[:, 0] += ox
    inners1[:, 1] += oy
    outers1[:, 1] += oy
    recon_img = np.zeros((256, 256), dtype=np.uint8)
    recon_lbl = np.zeros((256, 256), dtype=np.uint8)
    recon_patch = np.zeros((256, 256), dtype=np.uint8)
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
        piece = cv2.warpPerspective(target_patch[:, i * target_width_step:(i + 1) * target_width_step], np.linalg.inv(H), recon_patch.shape[1::-1])
        recon_patch = cv2.max(recon_patch, piece)
        piece = cv2.warpPerspective(target_img[:, i * target_width_step:(i + 1) * target_width_step], np.linalg.inv(H), recon_img.shape[1::-1])
        recon_img = cv2.max(recon_img, piece)
        piece = cv2.warpPerspective(target_lbl[:, i * target_width_step:(i + 1) * target_width_step], np.linalg.inv(H), recon_lbl.shape[1::-1])
        recon_lbl = cv2.max(recon_lbl, piece)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    recon_lbl_closed = cv2.morphologyEx(recon_lbl, cv2.MORPH_CLOSE, k)

    recon_img[recon_lbl_closed == 0] = 0
    recon_patch[recon_lbl_closed == 0] = 0

    stacked = np.hstack([recon_patch, recon_img, recon_lbl_closed])
    # cv2.imshow('stacked', stacked)
    # cv2.waitKey()

    fname = '%s_mold%d.png' % (n, idx_mold)
    cv2.imwrite(os.path.join(out_path, fname), stacked)
    print(fname, 'saved')
