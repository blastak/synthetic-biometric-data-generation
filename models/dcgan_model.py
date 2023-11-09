from models.base_GAN_model import *
from models.networks import *


class DCGANModel(BaseGANModel):
    def __init__(self, in_dims: int, out_channels: int, gpu_ids=[]):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.net_G = create_and_init(GeneratorDC(in_dims, out_channels), gpu_ids)
        self.net_D = create_and_init(Discriminator(out_channels, 'DC'), gpu_ids)
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.lossF_GAN = nn.BCELoss()

    def input_data(self, data):
        self.latent_vector = data['latent_vector'].to(self.device)
        self.real_image = data['real_image'].to(self.device)
        if self.fixed_data is None:
            self.fixed_data = self.latent_vector.clone()


    def forward(self):
        self.fake = self.net_G(self.latent_vector)

    def backward_G(self):
        set_requires_grad(self.net_D, False)

        self.optimizer_G.zero_grad()

        pred_fake = self.net_D(self.fake)
        self.loss_G = self.lossF_GAN(pred_fake, torch.tensor(1.).expand_as(pred_fake))
        self.loss_G.backward()

        self.optimizer_G.step()

    def backward_D(self):
        set_requires_grad(self.net_D, True)

        self.optimizer_D.zero_grad()

        pred_real = self.net_D(self.real_image)
        loss_GAN_real = self.lossF_GAN(pred_real, torch.tensor(1.).expand_as(pred_real))

        pred_fake = self.net_D(self.fake.detach())
        loss_GAN_fake = self.lossF_GAN(pred_fake, torch.tensor(0.).expand_as(pred_fake))

        self.loss_D = (loss_GAN_real + loss_GAN_fake) * 0.5
        self.loss_D.backward()

        self.optimizer_D.step()
