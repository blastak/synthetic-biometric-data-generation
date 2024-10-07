# resize, file save, rename

import argparse
from pathlib import Path

from PIL import Image

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', type=str, default=r'D:\Dataset\02_Iris\02_CASIA-IrisV4(JPG)\CASIA-Iris-Interval', help='original CASIA-IrisV4 data folder')
    ap.add_argument('--seg_map_in_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master\Data\CAV\guassian_noise_224', help='CASIA seg-map folder to read')
    ap.add_argument('--seg_map_out_dir', type=str, default=r'D:\Dataset\02_Iris\IrisSegment-master-extracted-CASIA', help='CASIA seg-map folder to save')
    opt = ap.parse_args()

    DATASET_DIR = opt.dataset_dir
    SEG_MAP_IN_DIR = opt.seg_map_in_dir
    SEG_MAP_OUT_DIR = opt.seg_map_out_dir

    w, h = (320, 280)

    # 이미지 리스트 불러오기
    segmap_paths = list(p.absolute() for p in Path(SEG_MAP_IN_DIR).glob('**/*.tiff'))
    assert len(segmap_paths) != 0, 'empty list'

    pd = Path(DATASET_DIR) # 파일 이름 참조를 위한 객체
    po = Path(SEG_MAP_OUT_DIR) # 저장 경로를 위한 객체

    for i,s in enumerate(segmap_paths):
        # original DB에서 파일 이름 검색
        jpgs = list(pd.glob(f'**/*{s.stem}.jpg'))
        assert len(jpgs) == 1, '없거나 두 개 이상임'
        j = jpgs[0]

        # 저장할 경로 생성
        sp = po / j.relative_to(j.parents[2]).with_suffix(s.suffix)
        sp.parent.mkdir(parents=True, exist_ok=True)

        # resize & save image
        img = Image.open(s).resize((w, h))
        # img.save(sp) # Uncomment to use

