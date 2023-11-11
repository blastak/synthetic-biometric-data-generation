import cv2
from torchvision.utils import make_grid

from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, choices=['fingerprint', 'iris', 'fingervein', 'handdorsalvein', 'handpalmvein'], help='modality name')
    parser.add_argument('--net_name', type=str, required=True, choices=['R_Thumbnail', 'R_Enhancement', 'D_IDPreserve'], help='Network name which will be trained')
    parser.add_argument('--data_dir', type=str, required=True, help='Absolute or relative path of input data directory for training')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size (default = xx)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads for data loading')
    args = parser.parse_args()

    gpu_ids = []
    for n in args.gpu_ids.split(','):
        if int(n) >= 0:
            gpu_ids.append(int(n))

    ########## torch environment settings
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
    torch.set_default_device(device)  # working on torch>2.0.0
    if torch.cuda.is_available() and len(gpu_ids) >= 1:
        torch.multiprocessing.set_start_method('spawn')

    ########## training dataset settings
    test_dataset = create_dataset(args.net_name, args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, generator=torch.Generator(device=device), num_workers=args.workers)

    ########## model settings
    if args.net_name == 'R_Thumbnail':
        model = create_model(args.net_name, 512, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    elif args.net_name == 'R_Enhancement':
        model = create_model(args.net_name, 1, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    elif args.net_name == 'D_IDPreserve':
        model = create_model(args.net_name, 2, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자

    #
    model.load_checkpoints(args.ckpt_path)

    ########## make saving folder
    d,f = os.path.split(args.ckpt_path)
    save_dir = os.path.join(d, f.split('.')[0])
    cnt = 1
    while True:
        try:
            os.makedirs(save_dir + '_test%03d' % cnt)
            save_dir += '_test%03d' % cnt
            break
        except:
            cnt += 1
    args.save_dir = save_dir
    save_log(save_dir, datetime.now(KST).strftime('%Y%m%d_%H%M%S%z'))
    save_log(save_dir, cvt_args2str(vars(args)))

    for i, inputs in enumerate(test_loader):
        model.input_data(inputs)
        model.testing()
        detached = model.fake_image.detach().cpu()
        montage = make_grid(detached, nrow=int(args.batch_size ** 0.5), normalize=True).permute(1, 2, 0).numpy()
        montage = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        save_image_path = os.path.join(save_dir, 'generated_image_%03d.jpg' % (i + 1))
        cv2.imwrite(save_image_path, montage)
        print(save_image_path + ' saved!')
        save_log(save_dir, save_image_path + ' saved!')

    save_log(save_dir, datetime.now(KST).strftime('%Y%m%d_%H%M%S%z'))
