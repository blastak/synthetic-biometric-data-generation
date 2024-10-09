"""
dataset_root 하위의 모든 bmp나 jpg를 읽어서, VeriEye로 subject(<-quality포함) 검출 후, subject와 pupil,iris polygon을 저장한다.
out_dir에 원본 데이터와 같은 폴더 구조로 subj, npz를 각각 저장한다.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from bio_modals.iris import Iris

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database', help='original data folder')
    ap.add_argument('--out_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database_iris_pupil', help='save folder')
    ap.add_argument('--SDK_dir', type=str, default=r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64', help='VeriEye SDK folder')
    opt = ap.parse_args()

    DATASET_ROOT = opt.dataset_root
    OUT_DIR = opt.out_dir
    SDK_DIR = opt.SDK_dir

    # 이미지 리스트 불러오기
    pd = Path(DATASET_ROOT)
    img_paths = list(p.absolute() for p in pd.glob('**/*') if p.suffix in ['.bmp', '.jpg'])
    assert len(img_paths) != 0, 'empty list'

    po = Path(OUT_DIR)  # 저장 경로를 위한 객체

    obj = Iris(SDK_DIR)
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        subject, quality = obj.create_subject(img)
        if subject is None:
            print('not detected', img_path)
            continue

        # saving instance of VeriEye's subject as '.subj'
        sp = po / img_path.relative_to(pd).with_suffix('.subj')
        sp.parent.mkdir(parents=True, exist_ok=True)
        obj.save_subject_template(sp.as_posix(), subject)

        # saving 32-sided polygons of pupil and iris as '.npz'
        bp = sp.with_suffix('.npz')
        att = subject.Irises.get_Item(0).Objects.get_Item(0)
        inners = [[inner.X, inner.Y] for inner in att.InnerBoundaryPoints]
        outers = [[outer.X, outer.Y] for outer in att.OuterBoundaryPoints]
        np.savez(bp.as_posix(), inners=np.array(inners), outers=np.array(outers))
