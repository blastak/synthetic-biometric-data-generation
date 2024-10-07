# resize, file save, rename

import argparse
from pathlib import Path

from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', type=str, default=r'D:\Dataset\02_Iris\01_IITD\IITD_Database', help='original IITD data folder')
    ap.add_argument('--seg_map_in_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master\Data\IITD\guassian_noise_224', help='IITD seg-map folder to read')
    ap.add_argument('--seg_map_out_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master-extracted-IITD', help='IITD seg-map folder to save')
    opt = ap.parse_args()

    DATASET_DIR = opt.dataset_dir
    SEG_MAP_IN_DIR = opt.seg_map_in_dir
    SEG_MAP_OUT_DIR = opt.seg_map_out_dir

    w, h = (320, 240)

    # 이미지 리스트 불러오기
    segmap_paths = list(p.absolute() for p in Path(SEG_MAP_IN_DIR).glob('**/*.tiff'))
    assert len(segmap_paths) != 0, 'empty list'

    pd = Path(DATASET_DIR) # 파일 이름 참조를 위한 객체
    po = Path(SEG_MAP_OUT_DIR) # 저장 경로를 위한 객체

    for i,s in enumerate(segmap_paths):
        # IrisSegment-master에서 파일명 분해하기
        # ex) s.stem: 'OperatorA_001-A_01'
        t = s.stem.split('-')
        f = t[0][-3:]
        n = t[1][-2:]

        # original DB에서 파일 이름 검색
        bmps = list(pd.glob(f'*{f}/{n}*.bmp'))
        assert len(bmps) == 1, '없거나 두 개 이상임'
        b = bmps[0]

        # 저장할 경로 생성
        # ex) po: D:/Dataset/02_Iris/IrisSegment-master-extracted-IITD
        # ex)  b: D:/Dataset/02_Iris/01_IITD/IITD_Database/001/01_L.bmp
        # ex) sp: D:/Dataset/02_Iris/IrisSegment-master-extracted-IITD/001/01_L.bmp
        sp = po / b.relative_to(b.parents[1]).with_suffix(s.suffix)
        sp.parent.mkdir(parents=True, exist_ok=True)

        # resize & save image
        img = Image.open(s).resize((w, h))
        # img.save(sp) # Uncomment to use

