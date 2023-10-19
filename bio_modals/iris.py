### built-in modules

### 3rd-party modules
from circle_fit import taubinSVD
from scipy.fftpack import fft, ifft, fftshift

### project modules
from neurotecbase import *


# https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer/38239574#38239574
# 사용 예: hr_round(angles_x,2) 는 소수점 셋째자리에서 반올림
def hr_round(val, digits=0):
    if digits == 0:
        return int(round(val + 10 ** (-len(str(val)) - 1), digits))
    return round(val + 10 ** (-len(str(val)) - 1), digits)


def gabor_convolve(im, nscale, minWaveLength, mult, sigmaOnf):
    rows, cols = im.shape
    filtersum = np.zeros(cols)

    EO = []  # Create an empty list to store results

    ndata = cols
    if ndata % 2 == 1:  # If there is an odd number of data points
        ndata = ndata - 1  # throw away the last one.

    logGabor = np.zeros(ndata)
    result = np.zeros((rows, ndata), dtype='complex_')
    # realpart = np.zeros((rows, ndata),dtype=bool)# 20230721 hrkim test
    # imagpart = np.zeros((rows, ndata),dtype=bool)# 20230721 hrkim test

    radius = np.arange((ndata // 2) + 1) / ((ndata // 2) * 2)  # Frequency values 0 - 0.5
    radius[0] = 1

    wavelength = minWaveLength  # Initialize filter wavelength.

    for s in range(nscale):  # For each scale.

        # Construct the filter - first calculate the radial filter component.
        fo = 1.0 / wavelength  # Centre frequency of filter.
        rfo = fo / 0.5  # Normalized radius from centre of frequency plane
        # corresponding to fo.
        logGabor[:ndata // 2 + 1] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[0] = 0

        filter = logGabor

        filtersum += filter

        # for each row of the input image, do the convolution, back transform
        for r in range(rows):  # For each row

            signal = im[r, :ndata]

            imagefft = fft(signal)

            result[r, :] = ifft(imagefft * filter)  # result[r, :] = (imagefft * filter)

            # 20230721 hrkim test  # mid = imagefft * filter  # realpart[r, :] = np.array(mid.real > 0, dtype=bool)  # imagpart[r, :] = np.array(mid.imag > 0, dtype=bool)

        # save the output for each scale
        EO.append(result.copy())

        wavelength *= mult  # Finally calculate Wavelength of next filter

    filtersum = fftshift(filtersum)

    return EO, filtersum


class Iris(NeurotecBase):
    def __init__(self, license_path=''):
        NeurotecBase.__init__(self, license_path)
        self.check_license('Biometrics.IrisExtraction,Biometrics.IrisMatching')

    def extract_feature(self, image):
        center = [0, 0]
        in_radius = 0
        out_radius = 0
        iris_code = np.empty([])

        # iris, pupil detection
        subject, quality = self.create_subject(image)
        if subject is None:
            return False, quality, iris_code, center, in_radius, out_radius

        inners = [[] for _ in range(32)]
        outers = [[] for _ in range(32)]
        for attr in subject.Irises.get_Item(0).Objects:
            for i, inner in enumerate(attr.InnerBoundaryPoints):
                inners[i] = [inner.X, inner.Y]
            for i, outer in enumerate(attr.OuterBoundaryPoints):
                outers[i] = [outer.X, outer.Y]
        inners = np.array(inners)
        outers = np.array(outers)

        # circle fitting
        xi, yi, ri, sigmai = taubinSVD(inners)
        xo, yo, ro, sigmao = taubinSVD(outers)
        center = [xi, yi]
        in_radius = ri
        out_radius = ro

        # C2P warping
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        polar_a = cv2.warpPolar(img_gray, (40 + 4, 240), (xi, yi), ro + 4, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar_b = cv2.rotate(polar_a, cv2.ROTATE_90_CLOCKWISE)

        # iris cropping
        G_y = np.array([-1, 0, 1]).reshape((3, 1))  # 3x1 필터 만들기
        filtered = cv2.filter2D(polar_b, cv2.CV_32F, G_y)  # 필터 적용
        proj = np.sum(filtered, axis=1)  # 세로축으로 프로젝션
        maxval = idx = 0
        ri_rect = hr_round(44 / (ro + 4) * ri)
        for i in range(ri_rect - 1, ri_rect + 2):
            if maxval < proj[i]:
                maxval = proj[i]
                idx = i
        idx += 1
        if idx + 20 > polar_b.shape[0]:
            idx = polar_b.shape[0] - 20
        E0, filtersum = gabor_convolve(polar_b[idx:, :], 1, 18, 1, 0.5)

        # thresholding with real_part, imaginary_part
        rp = (E0[0].real > 0).astype(np.uint8) * 255
        ip = (E0[0].imag > 0).astype(np.uint8) * 255
        iris_code = np.stack([rp, ip, ip], axis=2)

        return True, quality, iris_code, center, in_radius, out_radius

    def make_condition_image(self, feature_vector, position_angle_change: Optional[list] = None):
        pass

    def make_pair_image(self, image):
        pass

    def create_subject(self, img_or_file):
        subject = self.SDK.Biometrics.NSubject()
        iris = self.SDK.Biometrics.NIris()
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

        iris.Image = nimage
        subject.Irises.Add(iris)

        status = self.biometricClient.CreateTemplate(subject)
        quality = subject.GetTemplate().Irises.Records.get_Item(0).Quality
        if status != self.SDK.Biometrics.NBiometricStatus.Ok:
            return status, None, quality
        return status, subject, quality


import cv2

if __name__ == '__main__':
    obj = Iris(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')
    # img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp",cv2.IMREAD_GRAYSCALE)
    # obj.extract_feature(img1)

    ''' unit test of match() '''
    img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\10_L.bmp", cv2.IMREAD_GRAYSCALE)
    print(obj.match(img1, img2)) # case1: grayscale
    img1 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", cv2.IMREAD_COLOR)
    img2 = cv2.imread(r"D:\Dataset\IITD\IITD Database\001\10_L.bmp", cv2.IMREAD_COLOR)
    print(obj.match(img1, img2)) # case2: color
    img1 = r"D:\Dataset\IITD\IITD Database\001\01_L.bmp"
    img2 = r"D:\Dataset\IITD\IITD Database\001\10_L.bmp"
    print(obj.match(img1, img2)) # case3: path
    img1 = r'C:\weired\path'
    img2 = r'C:\weired\path2'
    try:
        print(obj.match(img1, img2)) # case4: exception
    except Exception as e:
        print(str(e))

    ''' unit test of match_using_filelist() '''
    filelist1 = [r"D:\Dataset\IITD\IITD Database\001\01_L.bmp", r"D:\Dataset\IITD\IITD Database\001\02_L.bmp", r"D:\Dataset\IITD\IITD Database\002\01_L.bmp"]
    filelist2 = [r"D:\Dataset\IITD\IITD Database\002\02_L.bmp", r"D:\Dataset\IITD\IITD Database\004\01_L.bmp", r"D:\Dataset\IITD\IITD Database\001\10_L.bmp"]
    imposter_match = obj.match_using_filelist(filelist1)
    print(imposter_match)
    genuine_match = obj.match_using_filelist(filelist1, filelist2)
    print(genuine_match)
