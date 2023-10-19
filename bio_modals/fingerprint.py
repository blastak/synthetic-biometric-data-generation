from base import *


class Fingerprint(Base):
    def __init__(self):
        Base.__init__(self)
        pass

    def extract_feature(self, image):
        pass

    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        pass

    def make_pair_image(self, image):
        pass

    def match(self, image1, image2):
        pass

    def match_bulk(self, filelist1, filelist2):
        pass
