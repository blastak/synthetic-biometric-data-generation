import argparse
import random
from datetime import datetime, timezone, timedelta

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import *
from utils import *

KST = timezone(timedelta(hours=9))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, choices=['fingerprint', 'iris', 'fingervein', 'handdorsalvein', 'handpalmvein'], help='modality name')
    parser.add_argument('--net_name', type=str, required=True, choices=['R_Thumbnail', 'R_Enhancement', 'D_IDPreserve'], help='Network name which will be trained')
    parser.add_argument('--data_dir', type=str, required=True, help='Absolute or relative path of input data directory for training')
    parser.add_argument('--exp_name', type=str, default='experiment_name', help='Output model name')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default = xxx)')
    parser.add_argument('--save_epochs', type=int, default=10, help='Frequency for saving checkpoints (in epochs) ')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size (default = xx)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads for data loading')
    args = parser.parse_args()

    gpu_ids = []
    for n in args.gpu_ids.split(','):
        if int(n) >= 0:
            gpu_ids.append(int(n))

    ########## torch environment settings
    manual_seed = 189649830
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
    torch.set_default_device(device)  # working on torch>2.0.0
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        torch.multiprocessing.set_start_method('spawn')

    ########## training dataset settings
    train_dataset = create_dataset(args.net_name, args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=args.workers)

    ########## model settings
    if args.net_name == 'R_Thumbnail':
        model = create_model(args.net_name, 512, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    elif args.net_name == 'R_Enhancement':
        model = create_model(args.net_name, 1, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    elif args.net_name == 'D_IDPreserve':
        model = create_model(args.net_name, 2, 1, gpu_ids)  # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    print(model)

    ########## make saving folder
    exp_name = args.exp_name
    if exp_name == 'experiment_name':
        exp_name = '_'.join([args.modality, args.net_name])
    save_dir = os.path.join('checkpoints', exp_name)
    cnt = 1
    while True:
        try:
            os.makedirs(save_dir + '_tr%03d' % cnt)
            save_dir += '_tr%03d' % cnt
            break
        except:
            cnt += 1
    args.exp_name = exp_name
    args.save_dir = save_dir
    save_log(save_dir, datetime.now(KST).strftime('%Y%m%d_%H%M%S%z'))
    save_log(save_dir, cvt_args2str(vars(args)))

    save_log(save_dir, '\n' + str(model) + '\n\n')

    ########## training process
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tq:
            for inputs in tq:
                model.input_data(inputs)
                model.learning()

                tq.set_description(f'Epoch {epoch}/{args.epochs}')
                tq.set_postfix(model.get_current_loss())
            save_log(save_dir, str(tq))

        if epoch % args.save_epochs == 0:
            ckpt_path = os.path.join(save_dir, 'ckpt_epoch%06d.pth' % epoch)
            model.save_checkpoints(ckpt_path)
            image_path = os.path.join(save_dir, 'image_epoch%06d.jpg' % epoch)
            model.save_generated_image(image_path)

    print('Finished training the model')
    print('checkpoints are saved in "%s"' % save_dir)
    save_log(save_dir, datetime.now(KST).strftime('%Y%m%d_%H%M%S%z'))
