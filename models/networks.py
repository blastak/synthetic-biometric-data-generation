import torch
import torch.nn as nn


class C0nvBnLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky_relu', conv_type='conv', filter_size=4, stride=2, padding=1):
        super().__init__()
        if conv_type == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, filter_size, stride, padding, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, filter_size, stride, padding, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GeneratorUnet(nn.Module):
    def __init__(self, in_channels):
        """
        GeneratorUnet
        :param in_channels: condition image의 채널수(EnhancementGAN의 경우 1, IDPreserveGAN의 경우 2~3)
        """
        super().__init__()
        inner_filters = 64
        nf = [inner_filters * 2 ** a for a in range(5)]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, nf[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder2 = C0nvBnLeaky(nf[0], nf[1])
        self.encoder3 = C0nvBnLeaky(nf[1], nf[2])
        self.encoder4 = C0nvBnLeaky(nf[2], nf[3])
        self.encoder5 = C0nvBnLeaky(nf[3], nf[3])  # 특이하게 512 -> 512
        self.encoder6 = C0nvBnLeaky(nf[3], nf[4])
        self.decoder6 = C0nvBnLeaky(nf[4], nf[3], 'relu', 'deconv')
        self.decoder5 = C0nvBnLeaky(nf[3] * 2, nf[3], 'relu', 'deconv')
        self.decoder4 = C0nvBnLeaky(nf[3] * 2, nf[2], 'relu', 'deconv')
        self.decoder3 = C0nvBnLeaky(nf[2] * 2, nf[1], 'relu', 'deconv')
        self.decoder2 = C0nvBnLeaky(nf[1] * 2, nf[0], 'relu', 'deconv')
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(nf[0] * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        d6 = self.decoder6(e6)
        d5 = self.decoder5(torch.cat((e5, d6), dim=1))
        d4 = self.decoder4(torch.cat((e4, d5), dim=1))
        d3 = self.decoder3(torch.cat((e3, d4), dim=1))
        d2 = self.decoder2(torch.cat((e2, d3), dim=1))
        d1 = self.decoder1(torch.cat((e1, d2), dim=1))
        return d1


class Discriminator(nn.Module):
    def __init__(self, in_channels, mode='patch'):
        """
        Discriminator
        :param in_channels: A,B가 concat된 총 채널수(EnhancementGAN의 경우 2, IDPreserveGAN의 경우 3~4, ThumbnailGAN은 1)
        :param mode: pix2pix용으로 쓸건지[default]. dcgan용으로 쓸건지
        """
        super().__init__()
        inner_filters = 64 if mode == 'patch' else 128
        nf = [inner_filters * 2 ** a for a in range(4)]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, nf[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = C0nvBnLeaky(nf[0], nf[1])
        self.layer3 = C0nvBnLeaky(nf[1], nf[2])
        if mode == 'patch':
            self.layer4 = C0nvBnLeaky(nf[2], nf[3], stride=1)  # stride=1
            self.layer5 = nn.Sequential(
                nn.Conv2d(nf[3], 1, 4, 1, 1, bias=False),  # stride=1
                nn.Sigmoid()
            )
        else:
            image_width = 64
            tensor_width = image_width // (2 ** 4) # 64->32->16->8->4
            self.layer4 = C0nvBnLeaky(nf[2], nf[3])
            self.layer5 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nf[3] * tensor_width ** 2, 1, bias=False),
                nn.Sigmoid()
            )  # dense

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class GeneratorDC(nn.Module):
    def __init__(self, in_dims, out_channels):
        """
        GeneratorDC
        :param in_dims: 512-dim random vector
        :param out_channels: output image channels
        """
        super().__init__()
        inner_filters = 128
        nf = [inner_filters * 2 ** a for a in range(4)]
        self.layer1 = nn.Sequential(
            nn.Linear(in_dims, nf[3] * 4 * 4, bias=False),
            nn.BatchNorm1d(nf[3] * 4 * 4),
            nn.ReLU(inplace=True)
        )  # dense
        self.unflatten = nn.Unflatten(1, (nf[3], 4, 4))
        self.layer2 = C0nvBnLeaky(nf[3], nf[2], 'relu', 'deconv')
        self.layer3 = C0nvBnLeaky(nf[2], nf[1], 'relu', 'deconv')
        self.layer4 = C0nvBnLeaky(nf[1], nf[0], 'relu', 'deconv')
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(nf[0], out_channels, 4, 2, 1, bias=False),
            nn.Tanh() # Binh의 논문과 다르게 BN은 넣지 않았음 (DCGAN에서는 안넣는듯해서)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.unflatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def create_and_init(net, gpu_ids=[]):
    if len(gpu_ids) > 1:
        net = nn.DataParallel(net, gpu_ids)
    net.apply(weights_init)
    return net
