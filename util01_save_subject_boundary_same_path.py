"""
이 파일은 Path 내의 모든 이미지 파일에서 VeriEye로 Iris를 검출하고, 같은 경로에 결과를 subj와 npz로 저장한다.
이미지를 자르거나 할때는 util02_save_cropimg_N_cropnpz_from_npz.py 를 사용하도록 한다
"""

import os
from pathlib import Path

import cv2
import numpy as np

from bio_modals.iris import Iris
from datasets import IMG_EXTENSIONS

obj = Iris(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')

# image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\IITD\IITD_Database').glob('**/*') if p.suffix == '.bmp')
# image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval').glob('**/*') if p.suffix == '.jpg')
image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\validation\images').glob('**/*') if p.suffix == '.png')

errors = []
for idx in range(len(image_path_list)):
    print(idx + 1, '/', len(image_path_list))
    img_np = cv2.imread(image_path_list[idx].as_posix(), cv2.IMREAD_GRAYSCALE)

    subject, quality = obj.create_subject(img_np)
    if subject is None:
        errors.append('subject is None %s' % image_path_list[idx])
        continue

    d, f = os.path.split(image_path_list[idx])
    n, e = os.path.splitext(f)
    obj.save_subject_template(os.path.join(d, n + '.subj'), subject)

    attrs = subject.Irises.get_Item(0).Objects
    inners = []
    outers = []
    for attr in attrs:
        for inner in attr.InnerBoundaryPoints:
            inners.append([inner.X, inner.Y])
        for outer in attr.OuterBoundaryPoints:
            outers.append([outer.X, outer.Y])
    inners = np.array(inners)
    outers = np.array(outers)

    np.savez(os.path.join(d, n + '.npz'), inners=inners, outers=outers)

print(errors)

### error 가 출력안됐을때 사용
# image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval').glob('**/*') if p.suffix == '.jpg')
# npz_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval').glob('**/*') if p.suffix == '.npz')
#
# idx_img = -1
# for idx in range(len(npz_path_list)):
#     d1, f1 = os.path.split(npz_path_list[idx])
#     n1, e1 = os.path.splitext(f1)
#
#     while True:
#         idx_img += 1
#         d0, f0 = os.path.split(image_path_list[idx_img])
#         n0, e0 = os.path.splitext(f0)
#
#         if n0 == n1:
#             break
#         print(image_path_list[idx_img])
