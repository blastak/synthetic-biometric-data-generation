### built-in modules
import os
import sys

### 3rd-party modules
import clr  # package name : pythonnet

### project modules
from bio_modals.base import *


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
        if not all([subject1, subject2]):
            matched = None
            matching_score = -1
        else:
            status = self.biometricClient.Verify(subject1, subject2)
            matched = True if status == self.SDK.Biometrics.NBiometricStatus.Ok else False
            matching_score = subject1.MatchingResults.get_Item(0).Score
        return matched, matching_score

    def match(self, image1, image2):
        subject1, quality1 = self.create_subject(image1)
        subject2, quality2 = self.create_subject(image2)
        matching_score = self.__get_matching_score(subject1, subject2)
        return matching_score, quality1, quality2

    def match_using_filelist(self, filelist1, filelist2=None):
        # mode=1 서로 다른 리스트끼리 비교 (중복성1 검증시 사용 가능)
        # mode=2 하나의 리스트에서 서로 비교 (중복성2 검증시 사용 가능)
        N = len(filelist1)
        mode = 2 if filelist2 is None else 1
        if mode == 1:
            M = len(filelist2)
        else:
            M = N
            filelist2 = filelist1

        subjects1 = [None] * N
        qualities1 = [None] * N
        for i in range(N):
            print('create_subject in filelist1 %d/%d' % (i + 1, N))
            subjects1[i], qualities1[i] = self.create_subject(filelist1[i])

        if mode == 1:
            subjects2 = [None] * M
            qualities2 = [None] * M
            for i in range(M):
                print('create_subject in filelist2 %d/%d' % (i + 1, M))
                subjects2[i], qualities2[i] = self.create_subject(filelist2[i])
        else:
            subjects2 = subjects1
            qualities2 = qualities1

        cnt = 0
        results = []  # path1 path2 is_matched score
        for i in range(N):
            d, f = os.path.split(filelist1[i])
            d = os.path.split(d)[-1]
            path1 = os.path.join(d, f)
            s = 0 if mode == 1 else i + 1
            for j in range(s, M):
                matched, matching_score = self.__get_matching_score(subjects1[i], subjects2[j])

                d, f = os.path.split(filelist2[j])
                d = os.path.split(d)[-1]
                path2 = os.path.join(d, f)
                line_txt = '%s %s %s %d' % (path1, path2, matched, matching_score)
                results.append(line_txt)

                cnt += 1
                print('%06d %s' % (cnt, line_txt))

        return results, qualities1, qualities2
