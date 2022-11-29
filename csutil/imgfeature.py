import cv2

"""
这个类用来方便地对图形进行配准操作
整合cv2中各种各样的配准方法，使讨厌的格式转换成为过去
"""

from csutil.util import kp_to_numpy
    
class ImageFeature:
    def __init__(self, img, featurename='sift', detectAndCompute=None, detector=None, computer=None, num_feature=None, mask=None, points_format=None):
        """
        :param img:(numpy)需要提取特征的图像。
        :param detector: (str or function)特征提取的方法，已实现的方法可以以字符串的方式传入，没有的话可以传入函数。
        :param computer: (str or function or None):描述子计算方法，同上，None代表用自定义的。
        :param num_feature: (int)特征点的数目，有些方法不支持。
        :mask : 图像的面具（掩码），一般为None
        """
        self.img = img
        self.featurename = featurename
        self.detectAndCompute = detectAndCompute
        self.detector = detector
        self.computer = computer
        self.num_feature = num_feature
        self.mask = mask
        self.points_format = points_format

    def _detect_and_compute(self):
        ## 可插入多种特征描述子 只需要添加名称，如'sift'
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




