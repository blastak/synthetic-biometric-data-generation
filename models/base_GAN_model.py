from abc import ABC, abstractmethod

import torch
from torchvision.utils import save_image

class BaseGANModel(ABC):
    def __init__(self):
        super().__init__()
        self.gpu_ids = []

    @abstractmethod
    def input_data(self, data):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward_G(self):
        pass

    @abstractmethod
    def backward_D(self):
        pass

    def learning(self):
        self.forward()
        self.backward_G()
        self.backward_D()

    def testing(self):
        with torch.no_grad():
            self.forward()

    def eval(self):
        self.__dict__['net_G'].eval()
        self.__dict__['net_D'].eval()

    def __repr__(self):
        return 'net_G:\n%s\n\nnet_D:\n%s' % (self.__dict__['net_G'], self.__dict__['net_D'])

    def get_current_loss(self):
        return {'loss_G': float(self.__dict__['loss_G']), 'loss_D': float(self.__dict__['loss_D'])}

    def save_checkpoints(self, ckpt_path):
        if isinstance(self.__dict__['net_G'], torch.nn.DataParallel):
            torch.save({
                'modelD_state_dict': self.__dict__['net_D'].module.cpu().state_dict(),
                'modelG_state_dict': self.__dict__['net_G'].module.cpu().state_dict(),
                'optimizerD_state_dict': self.__dict__['optimizer_D'].state_dict(),
                'optimizerG_state_dict': self.__dict__['optimizer_G'].state_dict(),
            }, ckpt_path)
            self.__dict__['net_D'].cuda(self.gpu_ids[0])
            self.__dict__['net_G'].cuda(self.gpu_ids[0])
        else:
            torch.save({
                'modelD_state_dict': self.__dict__['net_D'].state_dict(),
                'modelG_state_dict': self.__dict__['net_G'].state_dict(),
                'optimizerD_state_dict': self.__dict__['optimizer_D'].state_dict(),
                'optimizerG_state_dict': self.__dict__['optimizer_G'].state_dict(),
            }, ckpt_path)

    def save_generated_image(self, image_path):
        with torch.no_grad():
            img = self.__dict__['net_G'](self.__dict__['fixed']).detach().cpu()
        save_image(img, image_path, nrow=int(self.__dict__['fixed'].shape[0] ** 0.5), normalize=True)
