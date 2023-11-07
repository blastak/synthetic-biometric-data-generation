from models.dcgan_model import *


class ThumbnailGAN(DCGANModel):
    def __init__(self, in_dims: int, out_channels: int, gpu_ids=[]):
        super().__init__(in_dims, out_channels, gpu_ids)


# import os
# import random
# import argparse
#
# import cv2
# import numpy as np
# from tqdm import tqdm
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
#
#
# class ThumbnailGenerator(nn.Module):
#     def __init__(self):
#         super(ThumbnailGenerator, self).__init__()
#         self.fc = nn.Linear(512, 1024*4*4, bias=False)
#         self.bn1d = nn.BatchNorm1d(1024*4*4)
#         self.relu = nn.ReLU()
#         self.deconvs = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
#             nn.Tanh() # Binh의 논문과 다르게 BN은 넣지 않았음 (DCGAN에서는 안넣는듯해서)
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.bn1d(x)
#         x = self.relu(x)
#         x = x.reshape(x.shape[0], -1, 4, 4)
#         x = self.deconvs(x)
#         return x
#
#
# class ThumbnailDiscriminator(nn.Module):
#     def __init__(self):
#         super(ThumbnailDiscriminator, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 128, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--name', type=str, default='experiment_name', help='Output model name')
#     parser.add_argument('-tr', '--train-dir', type=str, default='train_dir', help='Input data directory for training')
#     parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs (default = xxx)')
#     parser.add_argument('-bs', '--batch-size', type=int, default=64, help='Mini-batch size (default = xx)')
#     parser.add_argument('-lr', '--learning-rate', type=float, default=0.0002, help='Learning rate (default = 0.0002)')
#     parser.add_argument('-se', '--save-epochs', type=int, default=10, help='Freqnecy for saving checkpoints (in epochs) ')
#     parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
#     parser.add_argument('--workers', type=int, default=2, help='Number of worker threads for data loading')
#     parser.add_argument('--display_on', action='store_true')
#     args = parser.parse_args()
#
#     experiment_name = args.name
#     train_dir = args.train_dir
#     num_epochs = args.epochs
#     batch_size = args.batch_size
#     learning_rate = args.learning_rate
#     save_epochs = args.save_epochs
#     display_on = args.display_on
#     workers = args.workers
#     gpu_ids = []
#     for s in args.gpu_ids.split(','):
#         if int(s) >= 0:
#             gpu_ids.append(int(s))
#
#     ########## torch environment settings
#     manual_seed = 999
#     random.seed(manual_seed)
#     torch.manual_seed(manual_seed)
#
#     device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
#     torch.set_default_device(device) # working on torch>2.0.0
#
#     ########## training dataset settings
#     image_size = 64
#     train_dataset = dset.ImageFolder(root=train_dir, transform=transforms.Compose([
#                                 transforms.Resize(image_size),
#                                 transforms.CenterCrop(image_size),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ## 이거 rgb 아닌가?
#                                 transforms.Grayscale(),
#                            ]))
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=workers)
#
#     ########## model settings
#     mymodel_D = ThumbnailDiscriminator()
#     mymodel_G = ThumbnailGenerator()
#     # mymodel_G.apply(weights_init)
#     # mymodel_D.apply(weights_init)
#     if device.type == 'cuda' and len(gpu_ids) > 1:
#         try:
#             torch.multiprocessing.set_start_method('spawn')
#             mymodel_D = nn.DataParallel(mymodel_D, gpu_ids)
#             mymodel_G = nn.DataParallel(mymodel_G, gpu_ids)
#         except Exception as e:
#             print('Exception!',str(e))
#             exit(1357)
#
#     ########## loss function & optimizer settings
#     bce_loss = nn.BCELoss()
#     optimizerD = optim.Adam(mymodel_D.parameters(), lr=learning_rate, betas=(0.5, 0.999)) # THB논문에서는 beta에 대한 언급이 없다. DCGAN을 따라 하자
#     optimizerG = optim.Adam(mymodel_G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
#
#     ########## make saving folder
#     experiment_dir = os.path.join('weights', experiment_name)
#     cnt = 1
#     while True:
#         try:
#             os.makedirs(experiment_dir + '_tr%03d' % cnt)
#             experiment_dir += '_tr%03d' % cnt
#             break
#         except:
#             cnt += 1
#
#     ########## training process
#     fixed_noise = torch.randn(batch_size, 512, device=device)
#     for epoch in range(1,num_epochs+1):
#         with tqdm(train_loader, unit='batch') as tq:
#             mymodel_G.train()
#             for inputs,_ in tq:
#                 inputs = inputs.to(device)
#                 ## Train with all-real batch : To maximize log(D(x))
#                 optimizerD.zero_grad()
#                 outputs = mymodel_D(inputs).view(-1)
#                 labels_real = torch.ones(outputs.shape[0], dtype=torch.float)
#                 loss_D_real = bce_loss(outputs, labels_real) # BCE_loss는 reduce_mean이 default이므로 값이 scalar로 출력된다
#                 loss_D_real.backward()
#
#                 ## Train with all-fake batch : To maximize log(1 - D(G(z)))
#                 noise = torch.randn(outputs.shape[0], 512)
#                 fake = mymodel_G(noise)
#                 outputs = mymodel_D(fake.detach()).view(-1) # 여기에서 G backward는 안하는거라서 detach함
#                 labels_fake = torch.zeros_like(labels_real)
#                 loss_D_fake = bce_loss(outputs, labels_fake)
#                 loss_D_fake.backward()
#                 ## update D
#                 optimizerD.step()
#
#                 ## Train with all-fake batch : To maximize log(D(G(z)))
#                 optimizerG.zero_grad()
#                 outputs = mymodel_D(fake).view(-1) # 생성을 다시 하지는 않고, 업데이트 된 D를 이용
#                 loss_G = bce_loss(outputs, labels_real) # 생성자의 손실값을 알기위해 라벨을 '진짜'라고 입력
#                 loss_G.backward()
#                 ## update G
#                 optimizerG.step()
#
#                 tq.set_description(f'Epoch {epoch}/{num_epochs}')
#                 tq.set_postfix(G_='%.4f'%loss_G.item(), D_real='%.4f'%loss_D_real.item(), D_fake='%.4f'%loss_D_fake.item())
#
#             if epoch % save_epochs == 0:
#                 ckpt_path = os.path.join(experiment_dir, 'ckpt_epoch%d.pth' % epoch)
#                 if isinstance(mymodel_G, nn.DataParallel):
#                     torch.save({
#                         'modelD_state_dict': mymodel_D.module.cpu().state_dict(),
#                         'modelG_state_dict': mymodel_G.module.cpu().state_dict(),
#                         'optimizerD_state_dict': optimizerD.state_dict(),
#                         'optimizerG_state_dict': optimizerG.state_dict(),
#                     },ckpt_path)
#                     mymodel_D.cuda(gpu_ids[0])
#                     mymodel_G.cuda(gpu_ids[0])
#                     ######## 아래는 load_state_dict할때 사용 예정
#                     # if isinstance(net, torch.nn.DataParallel):
#                     #     net = net.module
#                     # state_dict = torch.load(load_path, map_location=device))
#                     # net.load_state_dict(state_dict)
#                     ########
#                 else:
#                     torch.save({
#                         'modelD_state_dict': mymodel_D.state_dict(),
#                         'modelG_state_dict': mymodel_G.state_dict(),
#                         'optimizerD_state_dict': optimizerD.state_dict(),
#                         'optimizerG_state_dict': optimizerG.state_dict(),
#                     },ckpt_path)
#
#                 mymodel_G.eval()
#                 with torch.no_grad():
#                     img = mymodel_G(fixed_noise).detach().cpu()
#                     montage = make_grid(img, nrow=int(batch_size ** 0.5), normalize=True).permute(1,2,0).numpy()
#                     norm_image = cv2.normalize(montage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#                     norm_image = norm_image.astype(np.uint8)
#                     if display_on:
#                         cv2.imshow('big',norm_image)
#                         cv2.waitKey(1)
#                     filepath = os.path.join(experiment_dir, 'montage_%d.jpg' % epoch)
#                     cv2.imwrite(filepath,norm_image)
#
#     print('Finished training the model')
#     print('checkpoints are saved in "%s"' % experiment_dir)
