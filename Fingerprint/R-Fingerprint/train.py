import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

nc = 3
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(512, 1024*4*4, bias=False)
        self.bn1d = nn.BatchNorm1d(1024*4*4)
        self.relu = nn.ReLU()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh() # Binh의 논문과 다르게 BN은 넣지 않았음 (DCGAN에서는 안넣는듯해서)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1d(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.deconvs(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='experiment_name', help='Output model name')
    parser.add_argument('-tr', '--train-dir', type=str, default='train_dir', help='Input data directory for training')
    parser.add_argument('-e', '--epochs', type=int, default=2000, help='Number of epochs (default = 1500)')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='Mini-batch size (default = 64)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0002, help='Learning rate (default = 0.0002)')
    parser.add_argument('-se', '--save-epochs', type=int, default=100, help='Freqnecy for saving checkpoints (in epochs) ')
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    experiment_name = args.name
    train_dir = args.train_dir
    save_epochs = args.save_epochs

    experiment_dir = os.path.join('weights',experiment_name)
    cnt=1
    while True:
        try:
            os.makedirs(experiment_dir + '_tr%03d' % cnt)
            experiment_dir += '_tr%03d' % cnt
            break
        except:
            cnt+=1

    ########## torch environment settings
    manual_seed = 999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    torch.set_default_device(device)

    ########## training dataset settings
    image_size = 64
    train_dataset = dset.ImageFolder(root=train_dir, transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ## 이거 rgb 아닌가?
                                transforms.Grayscale(),
                           ]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))

    ########## model settings
    mymodel_G = Generator()
    mymodel_D = Discriminator()

    ########## loss function & optimizer settings
    bce_loss = nn.BCELoss()
    optimizerD = optim.Adam(mymodel_D.parameters(), lr=learning_rate, betas=(0.5, 0.999)) # THB논문에서는 beta에 대한 언급이 없다. DCGAN을 따라 하자
    optimizerG = optim.Adam(mymodel_G.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    ########## training process
    for epoch in range(num_epochs):
        total_loss_G = total_loss_D = 0.
        for i, (inputs,_) in enumerate(train_loader):
            inputs = inputs.to(device)
            ## Train with all-real batch : To maximize log(D(x))
            mymodel_D.zero_grad()
            outputs = mymodel_D(inputs).view(-1)
            labels_real = torch.ones(outputs.shape[0], dtype=torch.float)
            loss_D_real = bce_loss(outputs, labels_real)
            loss_D_real.backward() # BCE_loss는 reduce_mean이 default이므로 값이 scalar로 출력된다

            ## Train with all-fake batch : To maximize log(1 - D(G(z)))
            noise = torch.randn(outputs.shape[0], 512)
            fake = mymodel_G(noise)
            outputs = mymodel_D(fake.detach()).view(-1) # 여기에서 G backward는 안하는거라서 detach함
            labels_fake = torch.zeros_like(labels_real)
            loss_D_fake = bce_loss(outputs, labels_fake)
            loss_D_fake.backward()
            ## update D
            optimizerD.step()

            ## Train with all-fake batch : To maximize log(D(G(z)))
            mymodel_G.zero_grad()
            outputs = mymodel_D(fake).view(-1) # 생성을 다시 하지는 않고, 업데이트 된 D를 이용
            loss_G = bce_loss(outputs, labels_real) # 생성자의 손실값을 알기위해 라벨을 '진짜'라고 입력
            loss_G.backward()
            ## update G
            optimizerG.step()

            total_loss_D += loss_D_real.item() + loss_D_fake.item()
            total_loss_G += loss_G.item()
        total_loss_D = total_loss_D / len(train_loader)
        total_loss_G = total_loss_G / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} Loss(D): {total_loss_D:.4f} Loss(G): {total_loss_G:.4f}')

        if (epoch + 1) % save_epochs == 0:
            model_path_ckpt = os.path.join(experiment_dir, 'netG_epoch%d' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': mymodel_G.state_dict(),
                'optimizer_state_dict': optimizerG.state_dict()
            }, model_path_ckpt + '.pth')

            model_path_ckpt = os.path.join(experiment_dir, 'netD_epoch%d' % (epoch + 1))
            torch.save({
                'epoch': epoch,
                'model_state_dict': mymodel_D.state_dict(),
                'optimizer_state_dict': optimizerD.state_dict()
            }, model_path_ckpt + '.pth')

    # # Save the trained model
    # model_path_final = os.path.join(modeldir, netname)
    # torch.save({'model_state_dict': mymodel.state_dict()}, model_path_final + '.pth')
    # print('Saved model at:', model_path_final + '.pth')