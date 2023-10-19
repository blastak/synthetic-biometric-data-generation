from abc import ABC, abstractmethod
from typing import Optional


class Base(ABC):
    def __init__(self,license_path=''):
        pass

    @abstractmethod
    def extract_feature(self, image):
        pass

    @abstractmethod
    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        pass

    @abstractmethod
    def make_pair_image(self, image):
        pass

    @abstractmethod
    def match(self, obj1, obj2):
        pass
