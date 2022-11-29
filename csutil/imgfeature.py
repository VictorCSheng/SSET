import cv2

from csutil.util import kp_to_numpy
    
class ImageFeature:
    def __init__(self, img, featurename='sift', detectAndCompute=None, detector=None, computer=None, num_feature=None, mask=None, points_format=None):
        self.img = img
        self.featurename = featurename
        self.detectAndCompute = detectAndCompute
        self.detector = detector
        self.computer = computer
        self.num_feature = num_feature
        self.mask = mask
        self.points_format = points_format

    def _detect_and_compute(self):
        if self.featurename == 'sift':
            if self.num_feature is not None:
                self.feature = cv2.xfeatures2d.SIFT_create(nfeatures=self.num_feature)
            else:
                self.feature = cv2.xfeatures2d.SIFT_create()
                
            kp, des = self.feature.detectAndCompute(self.img, mask=self.mask)
        
        else:                
            if self.detectAndCompute != None: # 提取关键点和计算描述子
                kp, des = self.detectAndCompute(self.img, mask=self.mask)
            else:
                if self.detector != None:       # 用来提取关键点
                    kp = self.detector(self.img, mask=self.mask)
                else:
                    print('Feature point detection requires detector')
                    return 0
                if self.computer != None:       # 用来提取描述子
                    des = self.computer(self.img, kp)
                else:
                    des = None
            
        if self.points_format == 'numpy':
            kp = kp_to_numpy(kp)
            
        return kp, des




