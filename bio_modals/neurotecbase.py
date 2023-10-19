### built-in modules
import sys

### 3rd-party modules
import clr  # package name : pythonnet
import numpy as np

### project modules
from base import *



class NeurotecBase(Base):
    def __init__(self, license_path=''):
        Base.__init__(self)

        # license_path example : r"C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64" (do not use DLLs where dotNET folder)
        if license_path not in sys.path:
            sys.path.append(license_path)
            clr.AddReference('Neurotec')
            clr.AddReference('Neurotec.Biometrics')
            clr.AddReference('Neurotec.Biometrics.Client')
            clr.AddReference('Neurotec.Licensing')
            clr.AddReference('Neurotec.Media')
        self.SDK = __import__('Neurotec')
        self.is_activated = False
        self.biometricClient = self.SDK.Biometrics.Client.NBiometricClient()
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
    def create_subject(self, img_or_file):
        pass

    def check_license(self, modules_for_activating):
        if not self.is_activated:
            self.is_activated = self.SDK.Licensing.NLicense.ObtainComponents("/local", 5000, modules_for_activating)
            if not self.is_activated:
                exit(f'exit: no license {modules_for_activating}')
        return self.is_activated

    def __get_matching_score(self, subject1, subject2):
        if any([subject1, subject2]) is None:
            matching_score = -1
        elif self.biometricClient.Verify(subject1, subject2) != self.SDK.Biometrics.NBiometricStatus.Ok:
            matching_score = -1
        else:
            matching_score = subject1.MatchingResults.get_Item(0).Score
        return matching_score

    def match(self, image1, image2):
        ok1, subject1, quality1 = self.create_subject(image1)
        ok2, subject2, quality2 = self.create_subject(image2)
        matching_score = self.__get_matching_score(subject1, subject2)
        return matching_score, quality1, quality2

    def match_using_filelist(self, filelist1, filelist2=None):
        N = len(filelist1)
        mode = 'imposter' if filelist2 is None else 'genuine'
        if mode == 'imposter':
            M = N
            filelist2 = filelist1
        else:
            M = len(filelist2)

        scores = np.zeros([N, M], dtype=int)
        for i in range(N):
            ok1, subject1, quality1 = self.create_subject(filelist1[i])
            s = i + 1 if mode == 'imposter' else 0
            for j in range(s, M):
                ok2, subject2, quality2 = self.create_subject(filelist2[j])
                matching_score = self.__get_matching_score(subject1, subject2)
                scores[i, j] = matching_score
                print(filelist1[i], filelist2[j], matching_score)
        return scores
