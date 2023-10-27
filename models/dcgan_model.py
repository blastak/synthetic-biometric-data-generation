from models.base_model import *
from networks import *


class DCGANModel(BaseModel):
    def __init__(self, in_dims: int, out_channels: int, gpu_ids=[]):
        BaseModel.__init__(self)
        self.net_G = create_init(GeneratorDC(in_dims, out_channels), gpu_ids)
        self.net_D = create_init(Discriminator(out_channels, 'DC'), gpu_ids)
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.lossF_GAN = nn.BCELoss()

    def input_data(self, data):
        self.latent_vector = data['latent_vector']
        self.real_image = data['real_image']

    def forward(self):
        self.fake = self.net_G(self.latent_vector)

    def backward_G(self):
        self.optimizer_G.zero_grad()

        pred_fake = self.net_D(self.fake)
        self.loss_G = self.lossF_GAN(pred_fake, torch.tensor(1.).expand_as(pred_fake))
        self.loss_G.backward()

        self.optimizer_G.step()

    def backward_D(self):
        self.optimizer_D.zero_grad()

        pred_real = self.net_D(self.real_image)
        loss_GAN_real = self.lossF_GAN(pred_real, torch.tensor(1.).expand_as(pred_real))

        pred_fake = self.net_D(self.fake)
        loss_GAN_fake = self.lossF_GAN(pred_fake.detach(), torch.tensor(0.).expand_as(pred_fake))

        self.loss_D = (loss_GAN_real + loss_GAN_fake) * 0.5
        self.loss_D.backward()

        self.optimizer_D.step()
