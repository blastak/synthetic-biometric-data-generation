# No.05
"""
지문 impostor matching 해서 csv로 저장하는 프로그램
"""

import argparse
import csv
import os
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

real_or_synth = 'synth'

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\03_3_DIG_ALL_dpi500_subj", help='source subj folder')
    # ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\04_3_DIG_ALL_dpi500_10000(for_training)", help='source subj folder')
    ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\03_only_normal_dpi500_subj", help='source subj folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriFinger SDK folder')
    opt = ap.parse_args()

    SRC_SUBJ_DIR = opt.src_subj_dir
    SDK_DIR = opt.SDK_dir

    # subj 리스트 불러오기
    psrc_s = Path(SRC_SUBJ_DIR)
    subj_paths = list(p.absolute() for p in psrc_s.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths) != 0, 'empty list'

    adder = 1 if real_or_synth == 'real' else 0

    # subject 불러와서 id별로 dict에 저장
    obj = Fingerprint(SDK_DIR)
    D = {}
    for subj_path in subj_paths:
        subject = obj.load_subject_template(subj_path.as_posix())

        filename_tokens = subj_path.stem.split('_')

        id_fin = filename_tokens[0 + adder] + '_' + filename_tokens[2 + adder]  # ex) '00000003_R1'

        try:
            D[id_fin].append((subj_path, subject))
        except KeyError:
            D[id_fin] = [(subj_path, subject)]

    f = open(os.path.join(SRC_SUBJ_DIR, '__%s_vs_%s_impostor_matching.csv' % (real_or_synth, real_or_synth)), 'w', newline='')
    writer = csv.writer(f)
    L_keys = list(D.keys())
    for i in range(len(L_keys)):
        print(i + 1, '/', len(L_keys))
        for j in range(i + 1, len(L_keys)):
            subj_path1, subject1 = D[L_keys[i]][0]
            subj_path2, subject2 = D[L_keys[j]][0]
            is_match, score = obj.match_using_subjects(subject1, subject2)
            writer.writerow([subj_path1.stem, subj_path2.stem, is_match, score])
    f.close()
    print('total', len(D), 'ids')
