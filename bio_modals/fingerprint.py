### built-in modules

### 3rd-party modules

### project modules
from neurotecbase import *


class Fingerprint(NeurotecBase):
    def __init__(self, license_path=''):
        NeurotecBase.__init__(self, license_path)
        self.check_license('Biometrics.FingerExtraction,Biometrics.FingerMatching')

    def extract_feature(self, image):
        pass

    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        pass

    def make_pair_image(self, image):
        pass

    def create_subject(self, img_or_file):
        subject = self.SDK.Biometrics.NSubject()
        finger = self.SDK.Biometrics.NFinger()
        try:
            if type(img_or_file) == str:
                nimage = self.SDK.Images.NImage.FromFile(img_or_file)
            elif type(img_or_file) == np.ndarray:
                ww, hh = img_or_file.shape[1::-1]
                cc = 1
                if len(img_or_file.shape) == 3:
                    cc = img_or_file.shape[2]
                pixelformat = self.SDK.Images.NPixelFormat.Rgb8U if cc == 3 else self.SDK.Images.NPixelFormat.Grayscale8U
                nimage = self.SDK.Images.NImage.FromData(pixelformat, ww, hh, 0, ww * cc, self.SDK.IO.NBuffer.FromArray(img_or_file.tobytes()))
        except:
            raise TypeError('type is not supported')

        ''' code from Binh
        image.HorzResolution = 500
        image.VertResolution = 500
        image.ResolutionIsAspectRatio = False
        biometricClient.FingersTemplateSize = NTemplateSize.Small
        '''

        finger.Image = nimage
        subject.Fingers.Add(finger)

        status = self.biometricClient.CreateTemplate(subject)
        quality = subject.GetTemplate().Fingers.Records.get_Item(0).Quality
        if status != self.SDK.Biometrics.NBiometricStatus.Ok:
            return status, None, quality
        return status, subject, quality


import cv2

if __name__ == '__main__':
    obj = Fingerprint(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')
    # img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp",cv2.IMREAD_GRAYSCALE)
    # obj.extract_feature(img1)

    ''' unit test of match() '''
    # img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\10_L.bmp", cv2.IMREAD_GRAYSCALE)
    # print(obj.match(img1, img2)) # case1: grayscale
    # img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", cv2.IMREAD_COLOR)
    # img2 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\10_L.bmp", cv2.IMREAD_COLOR)
    # print(obj.match(img1, img2)) # case2: color
    # img1 = r"D:\Dataset\IITD\IITD Database\001\01_L.bmp"
    # img2 = r"D:\Dataset\IITD\IITD Database\001\10_L.bmp"
    # print(obj.match(img1, img2)) # case3: path
    # img1 = r'C:\weired\path'
    # img2 = r'C:\weired\path2'
    # try:
    #     print(obj.match(img1, img2)) # case4: exception
    # except Exception as e:
    #     print(str(e))

    ''' unit test of match_using_filelist() '''
    # filelist1 = [r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", r"D:\Dataset\IITD\IITD Database\001\02_L.bmp", r"D:\Dataset\IITD\IITD Database\002\01_L.bmp"]
    # filelist2 = [r"D:\Dataset\IITD\IITD Database\002\02_L.bmp", r"D:\Dataset\IITD\IITD Database\004\01_L.bmp", r"D:\Dataset\IITD\IITD Database\001\10_L.bmp"]
    # imposter_match = obj.match_using_filelist(filelist1)
    # print(imposter_match)
    # genuine_match = obj.match_using_filelist(filelist1, filelist2)
    # print(genuine_match)
