from models.pix2pix_model import *


class EnhancementGAN(Pix2pixModel):
    def __init__(self, in_channels: int, out_channels: int, gpu_ids=[]):
        super().__init__(in_channels, out_channels, gpu_ids)
