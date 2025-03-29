# No.02
"""
dataset_root 하위의 모든 bmp나 jpg를 읽어서, VeriFinger로 subject(<-quality포함) 검출 후, subject를 저장한다.
out_dir에 원본 데이터와 같은 폴더 구조로 subj를 저장한다.
quality로 내림차순 정렬해서 또 다른 폴더에 10000개만 저장해둔다
"""

import argparse
from pathlib import Path

from bio_modals.fingerprint import Fingerprint

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\only_normal_30000_dpi500", help='original data folder')
    ap.add_argument('--out_dir', type=str, default=r"E:\Dataset\05_Fingerprint\CVLabGenerated\only_normal_30000_dpi500_subj", help='save folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriFinger SDK folder')
    opt = ap.parse_args()

    DATASET_ROOT = opt.dataset_root
    OUT_DIR = opt.out_dir
    SDK_DIR = opt.SDK_dir

    # 이미지 리스트 불러오기
    pd = Path(DATASET_ROOT)
    img_paths = list(p.absolute() for p in pd.glob('**/*') if p.suffix.lower() in ['.bmp', '.jpg'])
    assert len(img_paths) != 0, 'empty list'

    po = Path(OUT_DIR)  # 저장 경로를 위한 객체

    obj = Fingerprint(SDK_DIR)
    for i, img_path in enumerate(img_paths):
        subject, quality = obj.create_subject(img_path)
        if subject is None:
            print('not detected', img_path)
            continue

        # saving instance of VeriFinger's subject as '.subj'
        sp = po / img_path.relative_to(pd).with_suffix('.subj')
        obj.save_subject_template(sp.as_posix(), subject)  # Uncomment to use
