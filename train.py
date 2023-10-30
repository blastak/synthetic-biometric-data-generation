# if torch.cuda.is_available() and len(gpu_ids) > 1:
#     try:
#         torch.multiprocessing.set_start_method('spawn')
###### 위에거 살려야함 set_default_device 다음에 어딘가 넣으면 되지 않을까?



import os
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import importlib
from datasets import *


def create_model(model_file_name, in_channels, out_channels, gpu_ids):
    lib = importlib.import_module('models.%s' % model_file_name)
    cls_name = model_file_name[2:] + 'GAN'
    if cls_name in lib.__dict__:
        model = lib.__dict__[cls_name]
        return model(in_channels, out_channels, gpu_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, required=True, choices=['fingerprint', 'iris', 'fingervein', 'handdorsalvein', 'handpalmvein'], help='modality name')
    parser.add_argument('--net_name', type=str, required=True, choices=['R_Thumbnail','R_Enhancement', 'D_IDPreserve'], help='Network name which will be trained')
    parser.add_argument('--data_dir', type=str, required=True, help='Absolute or relative path of input data directory for training')
    parser.add_argument('--exp_name', type=str, default='experiment_name', help='Output model name')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default = xxx)')
    parser.add_argument('--save_epochs', type=int, default=10, help='Freqnecy for saving checkpoints (in epochs) ')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size (default = xx)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker threads for data loading')
    parser.add_argument('--display_on', action='store_true')
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
    torch.set_default_device(device) # working on torch>2.0.0
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        torch.multiprocessing.set_start_method('spawn')


    # ########## training dataset settings
    train_dataset = ThumbnailDataset(args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=args.workers)

    ########## model settings
    model = create_model(args.net_name, 512, 1, gpu_ids) # 모달리티 별로 in, out 크기를 미리 설정해두고 사용하자
    print(model)

    ########## make saving folder
    experiment_dir = os.path.join('checkpoints', args.exp_name)
    cnt = 1
    while True:
        try:
            os.makedirs(experiment_dir + '_tr%03d' % cnt)
            experiment_dir += '_tr%03d' % cnt
            break
        except:
            cnt += 1

    # ########## training process
    for epoch in range(1, args.epochs + 1):
        with tqdm(train_loader, unit='batch') as tq:
            for inputs in tq:
                model.input_data(inputs)
                model.learning()

                tq.set_description(f'Epoch {epoch}/{args.epochs}')
                tq.set_postfix(model.get_current_loss())

        if epoch % args.save_epochs == 0:
            ckpt_path = os.path.join(experiment_dir, 'ckpt_epoch%06d.pth' % epoch)
            model.save_checkpoints(ckpt_path)
            image_path = os.path.join(experiment_dir, 'image_epoch%06d.png' % epoch)
            model.save_generated_image(image_path)

    print('Finished training the model')
    print('checkpoints are saved in "%s"' % experiment_dir)

    # fixed_noise = torch.randn(batch_size, 512, device=device)
    # for epoch in range(1,num_epochs+1):
    #     with tqdm(train_loader, unit='batch') as tq:
    #         mymodel_G.train()
    #         for inputs,_ in tq:
    #             inputs = inputs.to(device)
    #             ## Train with all-real batch : To maximize log(D(x))
    #             optimizerD.zero_grad()
    #             outputs = mymodel_D(inputs).view(-1)
    #             labels_real = torch.ones(outputs.shape[0], dtype=torch.float)
    #             loss_D_real = bce_loss(outputs, labels_real) # BCE_loss는 reduce_mean이 default이므로 값이 scalar로 출력된다
    #             loss_D_real.backward()
    #
    #             ## Train with all-fake batch : To maximize log(1 - D(G(z)))
    #             noise = torch.randn(outputs.shape[0], 512)
    #             fake = mymodel_G(noise)
    #             outputs = mymodel_D(fake.detach()).view(-1) # 여기에서 G backward는 안하는거라서 detach함
    #             labels_fake = torch.zeros_like(labels_real)
    #             loss_D_fake = bce_loss(outputs, labels_fake)
    #             loss_D_fake.backward()
    #             ## update D
    #             optimizerD.step()
    #
    #             ## Train with all-fake batch : To maximize log(D(G(z)))
    #             optimizerG.zero_grad()
    #             outputs = mymodel_D(fake).view(-1) # 생성을 다시 하지는 않고, 업데이트 된 D를 이용
    #             loss_G = bce_loss(outputs, labels_real) # 생성자의 손실값을 알기위해 라벨을 '진짜'라고 입력
    #             loss_G.backward()
    #             ## update G
    #             optimizerG.step()
    #
    #             tq.set_description(f'Epoch {epoch}/{num_epochs}')
    #             tq.set_postfix(G_='%.4f'%loss_G.item(), D_real='%.4f'%loss_D_real.item(), D_fake='%.4f'%loss_D_fake.item())
    #
    #         if epoch % save_epochs == 0:
    #             ckpt_path = os.path.join(experiment_dir, 'ckpt_epoch%d.pth' % epoch)
    #             if isinstance(mymodel_G, nn.DataParallel):
    #                 torch.save({
    #                     'modelD_state_dict': mymodel_D.module.cpu().state_dict(),
    #                     'modelG_state_dict': mymodel_G.module.cpu().state_dict(),
    #                     'optimizerD_state_dict': optimizerD.state_dict(),
    #                     'optimizerG_state_dict': optimizerG.state_dict(),
    #                 },ckpt_path)
    #                 mymodel_D.cuda(gpu_ids[0])
    #                 mymodel_G.cuda(gpu_ids[0])
    #                 ######## 아래는 load_state_dict할때 사용 예정
    #                 # if isinstance(net, torch.nn.DataParallel):
    #                 #     net = net.module
    #                 # state_dict = torch.load(load_path, map_location=device))
    #                 # net.load_state_dict(state_dict)
    #                 ########
    #             else:
    #                 torch.save({
    #                     'modelD_state_dict': mymodel_D.state_dict(),
    #                     'modelG_state_dict': mymodel_G.state_dict(),
    #                     'optimizerD_state_dict': optimizerD.state_dict(),
    #                     'optimizerG_state_dict': optimizerG.state_dict(),
    #                 },ckpt_path)
    #
    #             mymodel_G.eval()
    #             with torch.no_grad():
    #                 img = mymodel_G(fixed_noise).detach().cpu()
    #                 montage = make_grid(img, nrow=int(batch_size ** 0.5), normalize=True).permute(1,2,0).numpy()
    #                 norm_image = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #                 norm_image = norm_image.astype(np.uint8)
    #                 if display_on:
    #                     cv2.imshow('big',norm_image)
    #                     cv2.waitKey(1)
    #                 filepath = os.path.join(experiment_dir, 'montage_%d.jpg' % epoch)
    #                 cv2.imwrite(filepath,norm_image)
    #
    # print('Finished training the model')
    # print('checkpoints are saved in "%s"' % experiment_dir)
