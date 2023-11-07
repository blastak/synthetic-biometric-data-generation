from models.pix2pix_model import *


class IDPreserveGAN(Pix2pixModel):
    def __init__(self, in_channels: int, out_channels: int, gpu_ids=[]):
        super().__init__(in_channels, out_channels, gpu_ids)


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
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from torchvision.utils import make_grid
#
#
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
#     '.tif', '.TIF', '.tiff', '.TIFF',
# ]
#
# class DFingerprintDataset(Dataset):
#     def __init__(self, fingerprint_path, minutia_path):
#         self.minutia_path = minutia_path
#         self.fingerprint_path = fingerprint_path
#
#         self.listA = [f for f in os.listdir(minutia_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]
#         self.listB = [f for f in os.listdir(fingerprint_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]
#
#         tf = [transforms.ToTensor(), transforms.Resize((320, 320),antialias=True)]
#         self.tf_A = transforms.Compose(tf)
#         self.tf_B = transforms.Compose(tf + [transforms.Grayscale()])
#
#     def __len__(self):
#         return len(self.listA)
#
#     def __getitem__(self, index):
#         b,g,r = cv2.split(cv2.imread(os.path.join(self.minutia_path, self.listA[index])))
#         img_A = np.stack((b,g),2) # "g" is same as "r"
#
#         img_B = cv2.imread(os.path.join(self.fingerprint_path, self.listB[index]))
#
#         real_A = self.tf_A(img_A)
#         real_B = self.tf_B(img_B)
#
#         return real_A, real_B
#
#
# class ConvBatch(nn.Module):
#     def __init__(self, in_channels, out_channels, activation='leaky_relu', conv_type = 'conv', filter_size=4, stride=2, padding=1):
#         super(ConvBatch, self).__init__()
#         if conv_type == 'conv':
#             self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride, padding)
#         else:
#             self.conv = nn.ConvTranspose2d(in_channels, out_channels, filter_size, stride, padding)
#         if activation == 'leaky_relu':
#             self.act = nn.LeakyReLU(0.2)
#         else:
#             self.act = nn.ReLU()
#         self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
#
# class DFingerprintGenerator(nn.Module):
#     def __init__(self):
#         super(DFingerprintGenerator, self).__init__()
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(2, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2)
#         )
#         self.encoder2 = ConvBatch(64, 128)
#         self.encoder3 = ConvBatch(128, 256)
#         self.encoder4 = ConvBatch(256, 512)
#         self.encoder5 = ConvBatch(512, 512)
#         self.encoder6 = ConvBatch(512, 1024)
#         self.decoder6 = ConvBatch(1024, 512, 'relu', 'deconv')
#         self.decoder5 = ConvBatch(512*2, 512, 'relu', 'deconv')
#         self.decoder4 = ConvBatch(512*2, 256, 'relu', 'deconv')
#         self.decoder3 = ConvBatch(256*2, 128, 'relu', 'deconv')
#         self.decoder2 = ConvBatch(128*2, 64, 'relu', 'deconv')
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(64*2, 1, 4, 2, 1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#         e5 = self.encoder5(e4)
#         e6 = self.encoder6(e5)
#         d6 = self.decoder6(e6)
#         d5 = self.decoder5(torch.cat((e5,d6),1))
#         d4 = self.decoder4(torch.cat((e4,d5),1))
#         d3 = self.decoder3(torch.cat((e3,d4),1))
#         d2 = self.decoder2(torch.cat((e2,d3),1))
#         d1 = self.decoder1(torch.cat((e1,d2),1))
#         return d1
#
#
# class DFingerprintDiscriminator(nn.Module):
#     def __init__(self):
#         super(DFingerprintDiscriminator, self).__init__()
#         self.convbn1 = nn.Sequential(
#             nn.Conv2d(2+1, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2)
#         )
#         self.convbn2 = ConvBatch(64, 128)
#         self.convbn3 = ConvBatch(128, 256)
#         self.convbn4 = ConvBatch(256, 512, stride=1)
#         self.convbn5 = nn.Sequential(
#             nn.Conv2d(512, 1, 4, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.convbn1(x)
#         x = self.convbn2(x)
#         x = self.convbn3(x)
#         x = self.convbn4(x)
#         x = self.convbn5(x)
#         return x
#
#
# # ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.latent_vector, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.latent_vector, 1.0, 0.02)
#         nn.init.constant_(m.bias.latent_vector, 0)
#
# def set_requires_grad(nets, requires_grad=False):
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--name', type=str, default='experiment_name', help='Output model name')
#     parser.add_argument('--train-dir-minutia', type=str, default='minutia_dir', help='Input data directory for training')
#     parser.add_argument('--train-dir-fingerprint', type=str, default='fingerprint_dir', help='Input data directory for training')
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
#     train_dir_m = args.train_dir_minutia
#     train_dir_f = args.train_dir_fingerprint
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
#     device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
#     torch.set_default_device(device) # working on torch>2.0.0
#
#     ########## training dataset settings
#     image_size = 320
#     # train_dataset = dset.ImageFolder(root=train_dir, transform=transforms.Compose([
#     #                             transforms.Resize(image_size),
#     #                             transforms.CenterCrop(image_size),
#     #                             transforms.ToTensor(),
#     #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ## 이거 rgb 아닌가?
#     #                             transforms.Grayscale(),
#     #                        ]))
#     train_dataset = DFingerprintDataset(train_dir_f, train_dir_m)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device), num_workers=workers)
#
#     ########## model settings
#     mymodel_D = DFingerprintDiscriminator()
#     mymodel_G = DFingerprintGenerator()
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
#     lambda_L1 = 100.
#     l1_loss = nn.L1Loss()
#     gan_loss = nn.BCEWithLogitsLoss()
#     optimizerD = optim.Adam(mymodel_D.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
#     optimizerG = optim.Adam(mymodel_G.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=0.00001)
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
#     tf = transforms.Compose([transforms.Resize(64,antialias=True),transforms.Resize(256,antialias=True)])
#     for epoch in range(1,num_epochs+1):
#         fixed_input = {}
#         with tqdm(train_loader, unit='batch') as tq:
#             mymodel_G.train()
#             for inputsA, inputsB in tq:
#                 real_A = inputsA.to(device)
#                 real_B = inputsB.to(device)
#
#                 if len(fixed_input) == 0:
#                     fixed_input['real_A'] = real_A.detach()
#                     fixed_input['real_B'] = real_B.detach()
#
#                 ## forward
#                 fake_B = mymodel_G(real_A)
#
#                 ## update D
#                 set_requires_grad(mymodel_D, True)
#                 optimizerD.zero_grad()
#                 fake_AB = torch.cat((real_A, fake_B), 1) #fake
#                 pred_fake = mymodel_D(fake_AB.detach())
#                 loss_D_fake = gan_loss(pred_fake, torch.tensor(0.).expand_as(pred_fake))
#                 real_AB = torch.cat((real_A, real_B), 1) #real
#                 pred_real = mymodel_D(real_AB)
#                 loss_D_real = gan_loss(pred_real, torch.tensor(1.).expand_as(pred_real))
#                 loss_D = (loss_D_fake + loss_D_real) * 0.5
#                 loss_D.backward()
#                 optimizerD.step()
#
#                 ## update G
#                 set_requires_grad(mymodel_D, False)
#                 optimizerG.zero_grad()
#                 fake_AB = torch.cat((real_A, fake_B), 1)
#                 pred_fake = mymodel_D(fake_AB)
#                 loss_G_GAN = gan_loss(pred_fake, torch.tensor(1.).expand_as(pred_real))
#                 loss_G_L1 = l1_loss(fake_B, real_B)
#                 loss_G = loss_G_GAN + loss_G_L1 * lambda_L1
#                 loss_G.backward()
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
#                     img_fake_B = mymodel_G(fixed_input['real_A']).detach().cpu()
#                     montage_fake_B = make_grid(img_fake_B, nrow=int(batch_size ** 0.5), normalize=True).permute(1, 2, 0).numpy()
#                     montage_fake_B = cv2.normalize(montage_fake_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
#                     img_real_A = fixed_input['real_A'].cpu()
#                     montage_real_A = make_grid(img_real_A, nrow=int(batch_size ** 0.5), normalize=True).permute(1, 2, 0).numpy()
#                     montage_real_A = cv2.normalize(montage_real_A, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
#                     montage_real_A = np.stack((montage_real_A[:,:,0],montage_real_A[:,:,1],montage_real_A[:,:,1]),2)
#                     img_real_B = fixed_input['real_B'].cpu()
#                     montage_real_B = make_grid(img_real_B, nrow=int(batch_size ** 0.5), normalize=True).permute(1, 2, 0).numpy()
#                     montage_real_B = cv2.normalize(montage_real_B, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
#                     if display_on:
#                         cv2.imshow('condition images',montage_real_A)
#                         cv2.imshow('generated images',montage_fake_B)
#                         cv2.imshow('real images',montage_real_B)
#                         cv2.waitKey(1)
#                     cv2.imwrite(os.path.join(experiment_dir, '_%d_condition_image.jpg' % epoch),montage_real_A)
#                     cv2.imwrite(os.path.join(experiment_dir, '_%d_generated_image.jpg' % epoch),montage_fake_B)
#                     cv2.imwrite(os.path.join(experiment_dir, '_%d_real_image.jpg' % epoch),montage_real_B)
#
#     print('Finished training the model')
#     print('checkpoints are saved in "%s"' % experiment_dir)
