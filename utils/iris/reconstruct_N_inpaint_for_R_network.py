# No.04
"""
576x66 img로 i40+o125 256x256 이미지를 만들건데,
reconstruction 하고, 빈 부분을 inpainting 한 후에 256x256 저장
256x256 Iris-recon은 다음 두 가지 용도로 사용된다.
1) 64x64로 만들어서 Random-network의 학습
2) 256->64->256으로 만들어서 enhancement 학습
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def new_points32(img_size=256, inner_radius=50, outer_radius=128):
    inners = []
    outers = []
    cx = img_size // 2
    cy = img_size // 2
    for ang in np.arange(0, 360, 360 / 32):
        cval = np.cos(np.deg2rad(ang))
        sval = np.sin(np.deg2rad(ang))
        dx = round(cx + inner_radius * cval)
        dy = round(cy + inner_radius * sval)
        inners.append([dx, dy])
        dx = round(cx + outer_radius * cval)
        dy = round(cy + outer_radius * sval)
        outers.append([dx, dy])
    return np.array(inners), np.array(outers)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--iriscode_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iriscode', help='iris-code data folder')
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

    inners1, outers1 = new_points32(size_recon, inner_radius=40, outer_radius=125)
    recon_img1 = np.zeros([size_recon, size_recon], dtype=np.uint8)
    recon_segmap1 = np.zeros([size_recon, size_recon], dtype=np.uint8)

    for p, img_path in enumerate(img_paths):
        # 데이터 로드
        iris_code_img = np.uint8(Image.open(img_paths[p]))
        iris_code_segmap = np.uint8(Image.open(segmap_paths[p]))

        # 빈 이미지 준비
        recon_img = recon_img1.copy()
        recon_segmap = recon_segmap1.copy()
        for i in range(32):
            p_tl, p_bl = inners1[i], outers1[i]
            p_tr, p_br = inners1[(i + 1) % 32], outers1[(i + 1) % 32]
            pts = np.float32([p_tl, p_tr, p_br, p_bl])  # CW
            pts2 = np.float32([[0, 0], [piece_w, 0], [piece_w, piece_h], [0, piece_h]])
            invH = np.linalg.inv(cv2.getPerspectiveTransform(pts, pts2))
            piece = cv2.warpPerspective(iris_code_img[:, i * piece_w:(i + 1) * piece_w], invH, recon_img.shape[1::-1])
            recon_img = cv2.bitwise_or(recon_img, piece)
            piece = cv2.warpPerspective(iris_code_segmap[:, i * piece_w:(i + 1) * piece_w], invH, recon_segmap.shape[1::-1])
            recon_segmap = cv2.bitwise_or(recon_segmap, piece)
        # recon_img의 fill gap (using inpainting)
        mask_inpaint = np.zeros_like(recon_img)
        for i in range(len(inners1)):
            cv2.line(mask_inpaint, inners1[i], outers1[i], color=255, thickness=2)
        recon_img = cv2.inpaint(recon_img, mask_inpaint, 3, cv2.INPAINT_NS)
        # recon_segmap의 fill gap (using morph_close3x3)
        recon_segmap_closed = cv2.morphologyEx(recon_segmap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # refinement
        recon_img[recon_segmap_closed < 128] = 0  # segmap의 흰색 부분만 유효한 img의 픽셀임 (눈꺼풀 등 제거)
        cv2.circle(recon_img, (size_recon // 2, size_recon // 2), 40 - 2, color=0, thickness=cv2.FILLED)  # 동공(pupil) 부분 검은색 채우기
        # cv2.imshow('recon_img',recon_img)
        # cv2.waitKey()

        # file save
        out_img_p = po / img_path.relative_to(pic)
        out_img_p.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray(recon_img).save(out_img_p) # Uncomment to use
