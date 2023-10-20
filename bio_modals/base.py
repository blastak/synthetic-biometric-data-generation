from abc import ABC, abstractmethod
from typing import Optional


class Base(ABC):
    def __init__(self):
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
    def match(self, image1, image2):
        pass

    @abstractmethod
    def match_using_filelist(self, filelist1, filelist2=None):
        pass
