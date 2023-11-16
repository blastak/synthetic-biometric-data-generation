from models.dcgan_model import *


class ThumbnailGAN(DCGANModel):
    def __init__(self, in_dims: int, out_channels: int, gpu_ids=[]):
        super().__init__(in_dims, out_channels, gpu_ids)
