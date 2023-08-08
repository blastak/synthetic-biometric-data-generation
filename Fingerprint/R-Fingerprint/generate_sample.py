# import os
import random
import torch
import cv2

from train import Generator

if __name__=='__main__':
    # experiment_dir = os.path.join('weights','experiment_name_tr001')

    ########## torch environment settings
    # manual_seed = 999
    # random.seed(manual_seed)
    # torch.manual_seed(manual_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    torch.set_default_device(device)

    mymodel = Generator()
    mymodel.load_state_dict(torch.load('weights/experiment_name_tr001/netG_epoch2000.pth')['model_state_dict'])
    mymodel.to(device)

    mymodel.eval()

    for i in range(30):
        noise = torch.randn(1, 512)
        fake = mymodel(noise).detach().cpu()

        img = fake.numpy().squeeze()
        img = cv2.resize(img,(0,0),fx=4,fy=4)
        cv2.imshow('img',img)
        cv2.waitKey()