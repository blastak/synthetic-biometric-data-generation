# No.03
"""
image하고 subj를 quality 좋은순으로 10000개 뽑는 프로그램
각 ID별 손가락 개수 몇 개 인지도 저장할 것(csv:idnum, cnt)
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_img_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\02_3_DIG_ALL_dpi500", help='source image folder')
    ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\03_3_DIG_ALL_dpi500_subj", help='source subj folder')
    ap.add_argument('--dst_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\04_3_DIG_ALL_dpi500_10000", help='destination folder for both image and subj')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriEye SDK folder')
    opt = ap.parse_args()

    SRC_IMG_DIR = opt.src_img_dir
    SRC_SUBJ_DIR = opt.src_subj_dir
    DST_DIR = opt.dst_dir
    SDK_DIR = opt.SDK_dir

    # subj 리스트 불러오기
    psrc_s = Path(SRC_SUBJ_DIR)
    subj_paths = list(p.absolute() for p in psrc_s.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths) != 0, 'empty list'

    # subject 불러오고 quality 추출하고 리스트에 저장
    obj = Fingerprint(SDK_DIR)
    L = []
    for subj_path in subj_paths:
        subject = obj.load_subject_template(subj_path.as_posix())
        quality = obj.get_quality_from_subject(subject)
        L.append((subj_path, quality))

    # quality 순으로 10000개만 다른 폴더로 bmp복사와 subj복사
    psrc_i = Path(SRC_IMG_DIR)
    pdst = Path(DST_DIR)
    L2 = sorted(L, key=lambda l: l[1], reverse=True)[:10000]

    D = {}  # 'id':'sample개수'
    for src_subj_path, quality in L2:
        # copy image file into another folder
        src_img_path = psrc_i / (src_subj_path.stem + '.BMP')
        dst_img_path = pdst / (src_subj_path.stem + '.BMP')
        dst_subj_path = pdst / src_subj_path.name
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_subj_path, dst_subj_path)

        filename_tokens = src_subj_path.stem.split('_')
        id_fin = filename_tokens[1] + '_' + filename_tokens[3]  # ex) '00000003_R1'

        # 10000 장 이미지 안에 서로 다른 ID의 손가락이 몇 개인지 카운트 하기 위함
        try:
            D[id_fin] += 1
        except KeyError:
            D[id_fin] = 1

    with open(os.path.join(DST_DIR, '__count_fingers.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for k, v in D.items():
            writer.writerow([k, v])
