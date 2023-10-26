import torch
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass

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

    def train(self):
        self.forward()
        self.backward_G()
        self.backward_D()

    def test(self):
        with torch.no_grad():
            self.forward()

