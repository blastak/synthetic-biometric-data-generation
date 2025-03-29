# No.06
"""
지문 real vs synth impostor matching 해서 csv로 저장하는 프로그램
"""

import argparse
import csv
import os
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLab2004\03_3_DIG_ALL_dpi500_subj", help='real subj folder')
    ap.add_argument('--synth_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\03_only_normal_dpi500_subj", help='synth subj folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriFinger SDK folder')
    opt = ap.parse_args()

    REAL_SUBJ_DIR = opt.real_subj_dir
    SYNTH_SUBJ_DIR = opt.synth_subj_dir
    SDK_DIR = opt.SDK_dir

    # subj 리스트 불러오기 (real)
    p_real = Path(REAL_SUBJ_DIR)
    subj_paths_real = list(p.absolute() for p in p_real.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths_real) != 0, 'empty list'

    # subj 리스트 불러오기 (synth)
    p_synth = Path(SYNTH_SUBJ_DIR)
    subj_paths_synth = list(p.absolute() for p in p_synth.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths_synth) != 0, 'empty list'

    # subject 불러와서 id별로 dict에 저장
    obj = Fingerprint(SDK_DIR)
    D_real = {}
    for subj_path in subj_paths_real:
        filename_tokens = subj_path.stem.split('_')
        id_fin = filename_tokens[1] + '_' + filename_tokens[3]  # ex) '00000003_R1'

        if id_fin not in D_real:
            subject = obj.load_subject_template(subj_path.as_posix())
            D_real[id_fin] = [(subj_path, subject)]

    D_synth = {}
    for subj_path in subj_paths_synth:
        filename_tokens = subj_path.stem.split('_')
        id_fin = filename_tokens[0] + '_' + filename_tokens[2]  # ex) '00000003_R1'

        if id_fin not in D_synth:
            subject = obj.load_subject_template(subj_path.as_posix())
            D_synth[id_fin] = [(subj_path, subject)]

    f = open(os.path.join(SYNTH_SUBJ_DIR, '__real_vs_synth_impostor_matching.csv'), 'w', newline='')
    writer = csv.writer(f)
    L_real_keys = list(D_real.keys())
    L_synth_keys = list(D_synth.keys())
    for i in range(len(L_real_keys)):
        print(i + 1, '/', len(L_real_keys))
        subj_path1, subject1 = D_real[L_real_keys[i]][0]
        for j in range(len(L_synth_keys)):
            subj_path2, subject2 = D_synth[L_synth_keys[j]][0]
            is_match, score = obj.match_using_subjects(subject1, subject2)
            writer.writerow([subj_path1.stem, subj_path2.stem, is_match, score])
    f.close()
    print('total', len(D_real), 'x', len(D_synth), 'ids')
