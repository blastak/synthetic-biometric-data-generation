### built-in modules
from base import *
import sys

### 3rd-party modules
import clr  # package name : pythonnet


class Iris(Base):
    def __init__(self, license_path=''):
        # license_path example : r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64' (do not use DLLs where dotNET folder)
        Base.__init__(self, license_path)
        if license_path not in sys.path:
            sys.path.append(license_path)
            clr.AddReference('Neurotec')
            clr.AddReference('Neurotec.Biometrics')
            clr.AddReference('Neurotec.Biometrics.Client')
            clr.AddReference('Neurotec.Licensing')
            clr.AddReference('Neurotec.Media')
        self.SDK = __import__('Neurotec')

        # from Neurotec.Biometrics import NIris, NSubject, NTemplateSize, NBiometricStatus, NMatchingSpeed, NBiometricEngine, NTemplate, NFMinutiaFormat
        # from Neurotec.Biometrics.Client import NBiometricClient
        # from Neurotec.Licensing import NLicense
        # from Neurotec.Images import NImage, NPixelFormat
        # from Neurotec.IO import NBuffer

        self.is_activated = False
        self.biometricClient = self.SDK.Biometrics.Client.NBiometricClient()
        pass

    def __check_license_VeriEye(self):
        if not self.is_activated:
            modules_for_activating = "Biometrics.IrisExtraction,Biometrics.IrisMatching"
            self.is_activated = self.SDK.Licensing.NLicense.ObtainComponents("/local", 5000, modules_for_activating)
            if not self.is_activated:
                exit('exit: no VeriEye license')
        return self.is_activated

    def __create_subject_from_image(self, image):
        subject = self.SDK.Biometrics.NSubject()
        iris = self.SDK.Biometrics.NIris()
        # nimage = NImage.FromFile(filepath)
        nimage = self.SDK.Images.NImage.FromData(self.SDK.Images.NPixelFormat.Grayscale8U,
                                                 image.shape[1], image.shape[0], 0, image.shape[1],
                                                 self.SDK.IO.NBuffer.FromArray(image.tobytes()))
        iris.Image = nimage
        subject.Irises.Add(iris)

        if self.biometricClient.CreateTemplate(subject) != self.SDK.Biometrics.NBiometricStatus.Ok:
            return None, None
        quality = subject.GetTemplate().Irises.Records.get_Item(0).Quality
        return subject, quality

    def extract_feature(self, image):
        self.__check_license_VeriEye()
        subj1, quality = self.__create_subject_from_image(image)
        print(quality)
        pass

    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        # feature_vector는 iris code image, pos_ang_change 는 center(x,y), radius(inner,outer) 가 들어갈 수 있을 듯 하다
        self.__check_license_VeriEye()
        iris_code = feature_vector

        pass

    def make_pair_image(self, image):
        self.__check_license_VeriEye()
        feature_vector = self.extract_feature(image)
        img_condi = self.make_condition_image(feature_vector)

        pass

    def match(self, obj1, obj2):
        self.__check_license_VeriEye()
        pass


import cv2

if __name__ == '__main__':
    aa = Iris(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')
    img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp",cv2.IMREAD_GRAYSCALE)
    aa.extract_feature(img1)