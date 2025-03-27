# No.04
"""
지문 real vs. real genuine matching 해서 csv로 저장하는 프로그램
"""

import argparse
import csv
import os
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\03_3_DIG_ALL_dpi500_subj", help='source subj folder')
    # ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\04_3_DIG_ALL_dpi500_10000", help='source subj folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriEye SDK folder')
    opt = ap.parse_args()

    SRC_SUBJ_DIR = opt.src_subj_dir
    SDK_DIR = opt.SDK_dir

    # subj 리스트 불러오기
    psrc_s = Path(SRC_SUBJ_DIR)
    subj_paths = list(p.absolute() for p in psrc_s.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths) != 0, 'empty list'

    # subject 불러와서 id별로 dict에 저장
    obj = Fingerprint(SDK_DIR)
    D = {}
    for subj_path in subj_paths:
        subject = obj.load_subject_template(subj_path.as_posix())

        filename_tokens = subj_path.stem.split('_')
        id_fin = filename_tokens[1] + '_' + filename_tokens[3]  # ex) '00000003_R1'

        try:
            D[id_fin].append((subj_path, subject))
        except KeyError:
            D[id_fin] = [(subj_path, subject)]

    cnt = 0
    f = open(os.path.join(SRC_SUBJ_DIR, f'__real_vs_real_genuine_matching.csv'), 'w', newline='')
    writer = csv.writer(f)
    for key, L in D.items():
        if len(L) >= 2:
            cnt += 1
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                is_match, score = obj.match_using_subjects(L[i][1], L[j][1])
                writer.writerow([L[i][0].stem, L[j][0].stem, is_match, score])
    f.close()
    print('total', cnt, 'ids')
