from pathlib import Path

from bio_modals.iris import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

image_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\train\detected').glob('**/*') if p.suffix in IMG_EXTENSIONS)
label_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\train\detected').glob('**/*') if p.suffix == '.tiff')
npz_path_list = sorted(p.resolve() for p in Path(r'D:\Dataset\02_Iris\OpenEDS\Semantic_Segmentation_Dataset\train\detected').glob('**/*') if p.suffix == '.npz')

out_path = r'D:\Dataset\02_Iris\for_deterministic_ABM_color\train'
os.makedirs(out_path, exist_ok=True)

target_height = 66
target_width_step = 18
# idx_list = []
# while len(idx_list) < 500:
#     idx = random.randint(0,len(npz_path_list)-1)
#     if idx in idx_list:
#         continue
#     idx_list.append(idx)
for idx in range(len(npz_path_list)):
    print(idx + 1, '/', len(npz_path_list))

    d, f = os.path.split(npz_path_list[idx])
    n, e = os.path.splitext(f)

    img_np = cv2.imread(image_path_list[idx].as_posix(), cv2.IMREAD_GRAYSCALE)
    lbl_np = cv2.imread(label_path_list[idx].as_posix(), cv2.IMREAD_GRAYSCALE)
    hh, ww = img_np.shape[:2]

    loaded = np.load(npz_path_list[idx].as_posix())
    inners = loaded['inners']
    outers = loaded['outers']
    inners1 = np.roll(inners[::-1], 1, axis=0)  # 역순으로 바꾸기
    outers1 = np.roll(outers[::-1], 1, axis=0)

    target_img = np.empty([target_height, 0])
    for i in range(len(inners1)):
        p_tl = inners1[i]
        p_bl = outers1[i]
        try:
            p_tr = inners1[i + 1]
            p_br = outers1[i + 1]
        except:
            p_tr = inners1[0]
            p_br = outers1[0]
        pts = np.float32([p_tl, p_tr, p_br, p_bl])  # 반시계
        pts2 = np.float32([[0, 0], [target_width_step, 0], [target_width_step, target_height], [0, target_height]])
        H = cv2.getPerspectiveTransform(pts, pts2)
        piece = cv2.warpPerspective(img_np, H, [target_width_step, target_height])
        target_img = np.hstack([target_img, piece])
    target_img = target_img.astype(dtype=np.uint8)

    # expt #4
    E0, filtersum = gabor_convolve(target_img, 1, 18, 1, 0.5)
    magni = np.abs(E0[0])  # 여기서 abs는 complex number에서 magnitude를 구하는 것임
    magni = (magni - magni.min()) / (magni.max() - magni.min())
    magni = (magni * 255).astype(np.uint8)
    phase = np.angle(E0[0])
    phase = (phase - phase.min()) / (phase.max() - phase.min())
    phase = (phase * 255).astype(np.uint8)
    target_img2 = np.stack([magni, phase, phase], axis=2)

    ## 다시 도넛으로 만들기
    recon_img = np.zeros([*img_np.shape, 3], dtype=np.uint8)
    for i in range(len(inners1)):
        p_tl = inners1[i]
        p_bl = outers1[i]
        try:
            p_tr = inners1[i + 1]
            p_br = outers1[i + 1]
        except:
            p_tr = inners1[0]
            p_br = outers1[0]
        pts = np.float32([p_tl, p_tr, p_br, p_bl])  # 반시계
        pts2 = np.float32([[0, 0], [target_width_step, 0], [target_width_step, target_height], [0, target_height]])
        H = cv2.getPerspectiveTransform(pts, pts2)
        piece = cv2.warpPerspective(target_img2[:, i * target_width_step:(i + 1) * target_width_step], np.linalg.inv(H), recon_img.shape[1::-1])
        recon_img = cv2.bitwise_or(recon_img, piece)
    recon_img[lbl_np != 170, :] = 0

    # recon_img276 = cv2.resize(recon_img, (276, 276))
    # img_np276 = cv2.resize(img_np, (276, 276))
    # lbl_np276 = cv2.resize(lbl_np, (276, 276))
    #
    # cnt = 0
    # for i in [0,10,20]:
    #     for j in [0,10,20]:
    #         cnt+=1
    #         if cnt % 2 == 0:
    #             continue
    #         stacked = np.hstack([recon_img276[i:i+256,j:j+256], img_np276[i:i+256,j:j+256], lbl_np276[i:i+256,j:j+256]])
    #         # cv2.imshow('stacked', stacked)
    #         # cv2.waitKey()
    #
    #         fname = '%s_%d_%d.png' % (n, (20-i)//10, (20-j)//10)
    #         cv2.imwrite(os.path.join(out_path, fname), stacked)
    #         print(fname, 'saved')

    recon_img = cv2.resize(recon_img, (256, 256))
    img_np = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR), (256, 256))
    lbl_np = cv2.resize(cv2.cvtColor(lbl_np, cv2.COLOR_GRAY2BGR), (256, 256))
    stacked = np.hstack([recon_img, img_np, lbl_np])
    # cv2.imshow('stacked', stacked)
    # cv2.waitKey()

    fname = '%s.png' % (n)
    # cv2.imwrite(os.path.join(out_path, fname), stacked)
    print(fname, 'saved')
