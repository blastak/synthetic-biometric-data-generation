import os
import numpy as np
import cv2

opj = os.path.join

if __name__=='__main__':

    ####
    temp_file_path = r'E:\Dataset\05_Fingerprint\CVLab2004\01_3_DIG_ALL\3_00000039_DIG00_R3_01_B.BMP'
    aa = cv2.imread(temp_file_path)
    cv2.imshow('aa',aa)
    ####

    raw_file_path = r'E:\Dataset\05_Fingerprint\CVLab2004\01_3_DIG_ALL\3_00000039_DIG00_R3_01_B.RAW'
    h = 320
    w = 280
    with open(raw_file_path, 'rb') as f:
        arr = np.fromfile(f, dtype='uint8', sep='')
        img = np.reshape(arr[-h*w:], [h,w])
        # print(r)
        cv2.imshow('img', img)
        cv2.waitKey()
        # cv2.imwrite(opj(prefix,r[:-3]+'BMP'), img)

    # for i, prefix in enumerate(prefixes):
    #     L = os.listdir(prefix)
    #     imgorg = cv2.imread(opj(prefix,L[0]))
    #     h,w = imgorg.shape[:2]
    #
    #     L_raw = [l for l in L if l.lower().endswith('raw')]
    #     print(prefix, len(L_raw))
    #     for r in L_raw:
    #         with open(opj(prefix,r), 'rb') as f:
    #             arr = np.fromfile(f,dtype='uint8',sep='')
    #             img = np.reshape(arr[-h*w:],[h,w])
    #             # print(r)
    #             cv2.imshow('img',img)
    #             cv2.waitKey()
    #             # cv2.imwrite(opj(prefix,r[:-3]+'BMP'), img)
