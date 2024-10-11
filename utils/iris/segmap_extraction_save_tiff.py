# No.01
"""
https://github.com/wowotou1022/IrisSegment내에 segmap 데이터를 불러와서
IITD(또는 CASIA-IrisV4) 원본 데이터와 같은 크기로 다시 저장함
폴더 구조도 원본 데이터와 동일하게 유지
"""

import argparse
from pathlib import Path

from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # ap.add_argument('--dataset_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database', help='original data folder')
    # ap.add_argument('--seg_map_in_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master\Data\IITD\guassian_noise_224', help='seg-map folder to read')
    ap.add_argument('--dataset_dir', type=str, default=r'D:\Dataset\02_Iris\02_CASIA-IrisV4(JPG)\CASIA-Iris-Interval', help='original data folder')
    ap.add_argument('--seg_map_in_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master\Data\CAV\guassian_noise_224', help='seg-map folder to read')
    ap.add_argument('--seg_map_out_dir', type=str, help='seg-map folder to save')
    opt = ap.parse_args()

    DATASET_DIR = opt.dataset_dir
    SEG_MAP_IN_DIR = opt.seg_map_in_dir
    SEG_MAP_OUT_DIR = DATASET_DIR + '_segmap' if opt.seg_map_out_dir is None else opt.seg_map_out_dir

    d_type = 'IITD'
    w, h = (320, 240)
    if 'casia' in DATASET_DIR.lower() and 'cav' in SEG_MAP_IN_DIR.lower():
        d_type = 'CASIA'
        w, h = (320, 280)

    # segmap 이미지 리스트 불러오기
    segmap_paths = list(p.absolute() for p in Path(SEG_MAP_IN_DIR).glob('**/*.tiff'))
    assert len(segmap_paths) != 0, 'empty list'

    pd = Path(DATASET_DIR)  # 파일 이름 참조를 위한 객체
    po = Path(SEG_MAP_OUT_DIR)  # 저장 경로를 위한 객체

    for i, s in enumerate(segmap_paths):
        leaf_name = s.stem + '.jpg'
        if d_type == 'IITD':
            # IrisSegment-master에서 파일명 분해하기 (IITD의 경우)
            # ex) s.stem: 'OperatorA_001-A_01'
            t = s.stem.split('-')
            f = t[0][-3:]
            n = t[1][-2:]
            leaf_name = f + '/' + n + '*.bmp'

        # original DB에서 파일 이름 검색
        refs = list(pd.glob(f'**/*{leaf_name}'))
        assert len(refs) == 1, '없거나 두 개 이상임'
        ref = refs[0]

        # 저장할 경로 생성
        # ex)  po: D:\Dataset\02_Iris\01_IITD\IITD_Database_segmap
        # ex) ref: D:\Dataset\02_Iris\01_IITD\IITD_Database\001\01_L.bmp
        # ex)  sp: D:\Dataset\02_Iris\01_IITD\IITD_Database_segmap\001\01_L.bmp
        sp = po / ref.relative_to(pd).with_suffix(s.suffix)
        sp.parent.mkdir(parents=True, exist_ok=True)

        # resize & save image
        img = Image.open(s).resize((w, h))
        # img.save(sp) # Uncomment to use
