# import numpy as np
# from pathlib import Path
#
# # npz_path = r'D:\Dataset\02_Iris\IITD\IITD_Database'
# # npz_path = r'D:\Dataset\02_Iris\CASIA-IrisV4(JPG)\CASIA-Iris-Interval'
# npz_path = r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\test\images'
# npz_path_list = sorted(p.resolve() for p in Path(npz_path).glob('**/*') if p.suffix == '.npz')
#
# sum_dist_inner_outer = 0
# sum_perimeter = 0
# sum_diameter = 0
# for i in range(len(npz_path_list)):
#     loaded = np.load(npz_path_list[i].as_posix())
#     inners = loaded['inners']
#     outers = loaded['outers']
#
#     dist_inner_outer = 0
#     perimeter = 0
#     diameter = 0
#     diff = inners-outers
#     for j in range(32):
#         perimeter += np.linalg.norm(outers[j-1]-outers[j])
#         dist_inner_outer += np.linalg.norm(diff[j])
#     diameter = max(outers[:,1])-min(outers[:,1])  # 그냥 가로만 체크한다
#
#     sum_diameter += diameter
#     sum_perimeter += perimeter
#     sum_dist_inner_outer += dist_inner_outer
#
# avg_dist_inner_outer = (sum_dist_inner_outer/32)/len(npz_path_list)
# avg_perimeter = sum_perimeter/len(npz_path_list)
# avg_diameter = sum_diameter/len(npz_path_list)
#
# print('avg_dist_inner_outer',avg_dist_inner_outer)
# print('avg_perimeter',avg_perimeter)
# print('avg_diameter',avg_diameter)


import numpy as np
from pathlib import Path
import cv2
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# npz_path = r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\validation\detected'
npz_path = r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\validation\images'
npz_path_list = sorted(p.resolve() for p in Path(npz_path).glob('**/*') if p.suffix == '.npz')

def disp_color(img,inners,outers):
    img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_i = (0, 0, 255)
    color_o = (0, 255, 0)
    for i in range(len(inners)):
        cv2.line(img_disp, inners[i-1], inners[i], color_i, 1)
        cv2.line(img_disp, outers[i-1], outers[i], color_o, 1)
        # cv2.line(img_disp1, inners[i], outers[i], color_conn, 1)
        # if i == 0:
        #     cv2.line(img_disp1, inners[i], outers[i], (0,0,255), 2)
        # elif i == 1:
        #     cv2.line(img_disp1, inners[i], outers[i], (0,128,255), 2)
        # elif i == 2:
        #     cv2.line(img_disp1, inners[i], outers[i], (0,255,255), 2)
        # elif i == 3:
        #     cv2.line(img_disp1, inners[i], outers[i], (0,255,0), 2)
        # elif i == 4:
        #     cv2.line(img_disp1, inners[i], outers[i], (255,0,0), 2)
    return img_disp

cnt=0
for idx in range(len(npz_path_list)):
    print(idx + 1, '/', len(npz_path_list))
    d, f = os.path.split(npz_path_list[idx])
    n, e = os.path.splitext(f)

    ####### npz가 있는데 tiff에 255가 매우 희박하거나 없으면 npz,subj 지우기
    # lbl_np = cv2.imread(os.path.join(d,n+'.tiff'), cv2.IMREAD_GRAYSCALE)
    # area_lbl = np.count_nonzero(lbl_np==255)
    # if area_lbl <= 1000:
    #     cnt+=1
    #     print(n)
    #     loaded = np.load(npz_path_list[idx].as_posix())
    #     inners = loaded['inners']
    #     outers = loaded['outers']
    #     img_np = cv2.imread(os.path.join(d, n + '.png'), cv2.IMREAD_GRAYSCALE)
    #     img_np_disp = disp_color(img_np,inners,outers)
    #     cv2.imshow('img_np_disp',img_np_disp)
    #     lbl_np_disp = disp_color(lbl_np,inners,outers)
    #     cv2.imshow('lbl_np_disp',lbl_np_disp)
    #     cv2.waitKey()

    ###### inner의 내부의 흰색 비율
    # lbl_np = cv2.imread(os.path.join(d,n+'.tiff'), cv2.IMREAD_GRAYSCALE)
    # loaded = np.load(npz_path_list[idx].as_posix())
    # inners = loaded['inners']
    # outers = loaded['outers']
    # mask = np.zeros_like(lbl_np)
    # cv2.drawContours(mask, [inners], -1, (255,255,255), -1)
    # num255_mask = cv2.countNonZero(mask)
    # crop = cv2.bitwise_and(lbl_np, mask)
    # num255_crop = np.count_nonzero(crop==255)
    # ratio = abs(num255_crop / num255_mask - 1)
    # # if ratio > 0.3:
    # if 0.2 <= ratio <= 0.3:  # 이걸이용하면 조금 가려진거 찾을 수 있음 10개 정도
    #     cnt+=1
    #     print(n)
    #     img_np = cv2.imread(os.path.join(d, n + '.png'), cv2.IMREAD_GRAYSCALE)
    #     img_np_disp = disp_color(img_np,inners,outers)
    #     cv2.imshow('img_np_disp',img_np_disp)
    #     lbl_np_disp = disp_color(lbl_np,inners,outers)
    #     cv2.imshow('lbl_np_disp',lbl_np_disp)
    #     cv2.waitKey()

    ######## outer 내부의 sclera 비율
    lbl_np = cv2.imread(os.path.join(d,n+'.tiff'), cv2.IMREAD_GRAYSCALE)
    loaded = np.load(npz_path_list[idx].as_posix())
    inners = loaded['inners']
    outers = loaded['outers']
    mask = np.zeros_like(lbl_np)
    cv2.drawContours(mask, [outers], -1, (255,255,255), -1)
    num_mask = cv2.countNonZero(mask)

    crop = cv2.bitwise_and(lbl_np, mask)
    num85_crop = np.count_nonzero(crop==85)
    if num85_crop > 1500:
        cnt+=1
        print(n)
        img_np = cv2.imread(os.path.join(d, n + '.png'), cv2.IMREAD_GRAYSCALE)
        img_np_disp = disp_color(img_np,inners,outers)
        cv2.imshow('img_np_disp',img_np_disp)
        lbl_np_disp = disp_color(lbl_np,inners,outers)
        cv2.imshow('lbl_np_disp',lbl_np_disp)
        cv2.waitKey()




    # diameter = max(inners[:,1])-min(inners[:,1])
    # radius = diameter/2
    # area = np.pi * radius**2
    #
    # lbl_np = cv2.imread(label_path_list[idx].as_posix(),cv2.IMREAD_GRAYSCALE)
    # img_np = cv2.imread(image_path_list[idx].as_posix(),cv2.IMREAD_GRAYSCALE)
    # area_lbl = np.count_nonzero(lbl_np==255)
    #
    # ratio = abs(area_lbl / area - 1)
    # print(image_path_list[idx].as_posix(), '%.4f' % ratio)
    #
    # if ratio > 0.2:
    #     cnt+=1
    #     img_disp1 = cv2.cvtColor(img_np,cv2.COLOR_GRAY2BGR)
    #     color_i = (0, 0, 255)
    #     color_o = (0, 255, 0)
    #     color_conn = (0, 255, 255)
    #     for i in range(-1, len(inners) - 1):
    #         cv2.line(img_disp1, inners[i], inners[i + 1], color_i, 2)
    #         cv2.line(img_disp1, outers[i], outers[i + 1], color_o, 2)
    #         # cv2.line(img_disp1, inners[i], outers[i], color_conn, 1)
    #         # if i == 0:
    #         #     cv2.line(img_disp1, inners[i], outers[i], (0,0,255), 2)
    #         # elif i == 1:
    #         #     cv2.line(img_disp1, inners[i], outers[i], (0,128,255), 2)
    #         # elif i == 2:
    #         #     cv2.line(img_disp1, inners[i], outers[i], (0,255,255), 2)
    #         # elif i == 3:
    #         #     cv2.line(img_disp1, inners[i], outers[i], (0,255,0), 2)
    #         # elif i == 4:
    #         #     cv2.line(img_disp1, inners[i], outers[i], (255,0,0), 2)
    #
    #     img_disp2 = cv2.cvtColor(lbl_np, cv2.COLOR_GRAY2BGR)
    #     color_i = (0, 0, 255)
    #     color_o = (0, 255, 0)
    #     color_conn = (0, 255, 255)
    #     for i in range(-1, len(inners) - 1):
    #         cv2.line(img_disp2, inners[i], inners[i + 1], color_i, 2)
    #         cv2.line(img_disp2, outers[i], outers[i + 1], color_o, 2)
    #     cv2.imshow('img_disp1', img_disp1)
    #     cv2.imshow('img_disp2', img_disp2)
    #     # cv2.imshow('lbl_np',lbl_np)
    #     cv2.waitKey()

print(cnt)