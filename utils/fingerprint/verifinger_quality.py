import argparse
import csv
import os
import shutil
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

from natsort import natsorted

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_subj_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\03_only_normal_30000_dpi500_subj", help='source subj folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriEye SDK folder')
    opt = ap.parse_args()

    SRC_SUBJ_DIR = opt.src_subj_dir
    SDK_DIR = opt.SDK_dir

    # subj 리스트 불러오기
    psrc_s = Path(SRC_SUBJ_DIR)
    subj_paths = natsorted(p.absolute() for p in psrc_s.glob('**/*') if p.suffix.lower() in ['.subj'])
    assert len(subj_paths) != 0, 'empty list'

    # subject 불러오고 quality 추출하고 리스트에 저장
    obj = Fingerprint(SDK_DIR)
    L = []
    for subj_path in subj_paths:
        subject = obj.load_subject_template(subj_path.as_posix())
        quality = obj.get_quality_from_subject(subject)
        L.append((subj_path.stem, quality))

    with open(os.path.join(SRC_SUBJ_DIR, '__file&quality.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(L)
