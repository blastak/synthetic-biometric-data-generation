import cv2

from bio_modals.neurotecbase import *


class Face(NeurotecBase):
    def __init__(self, library_path=''):
        NeurotecBase.__init__(self, library_path)
        self.check_license('Biometrics.FaceExtraction,Biometrics.FaceMatching')

    def extract_feature(self, img_or_subject):
        pass

    def make_condition_image(self, feature_vector, position_angle_change: Optional[dict] = None):
        pass

    def make_pair_image(self, image):
        pass


def unit_test_match(obj):
    print('######################### Unit Test 1 - grayscale image input #########################')
    img1 = cv2.imread("../unit_test_data/Face/you01.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../unit_test_data/Face/you02.jpg", cv2.IMREAD_GRAYSCALE)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 2 - color image input #########################')
    img1 = cv2.imread("../unit_test_data/Face/kang01.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../unit_test_data/Face/kang02.jpg", cv2.IMREAD_COLOR)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 3 - file path input #########################')
    img1 = "../unit_test_data/Face/kang01.jpg"
    img2 = "../unit_test_data/Face/kang02.jpg"
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 4 - weired path input #########################')
    img1 = r'C:\weired\path'
    img2 = r'C:\weired\path2'
    try:
        matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
        print(matched, matching_score, quality1, quality2)
    except Exception as e:
        print(type(e))
    print()


def unit_test_match_using_filelist(obj):
    print('######################### Unit Test 1 - filelist1 #########################')
    filelist1 = [
        "../unit_test_data/Face/you01.jpg",
        "../unit_test_data/Face/kang01.jpg",
        "../unit_test_data/Face/jung01.jpg",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1)
    print(results, qualities1, qualities2)

    print('################### Unit Test 2 - filelist1 and filelist2 ###################')
    filelist2 = [
        "../unit_test_data/Face/kang02.jpg",
        "../unit_test_data/Face/you02.jpg",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1, filelist2)
    print(results, qualities1, qualities2)

    print('######################### Unit Test 3 - file error #########################')
    filelist3 = [r"C:\weired\path", r"C:\weired\path2"]
    try:
        results, qualities1, qualities2 = obj.match_using_filelist(filelist3)
        print(results, qualities1, qualities2)
    except Exception as e:
        print(type(e))
    print()


if __name__ == '__main__':
    obj = Face(r'C:\Neurotec_Biometric_13_1_SDK\Bin\Win64_x64')

    unit_test_match(obj)
    unit_test_match_using_filelist(obj)
