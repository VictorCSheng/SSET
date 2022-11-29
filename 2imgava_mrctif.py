import os
import cv2
import numpy as np

def AvergeImg(imgdir):
    imgnamelist = os.listdir(imgdir)
    imgsum = 0
    for i in range(len(imgnamelist)):
        img_ori = cv2.imread(imgdir + imgnamelist[i])
        if img_ori.ndim == 3:
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)

        imgsum = imgsum + img_ori.astype(float)

    imgavg = imgsum / len(imgnamelist)

    return imgavg.astype(np.uint8)

if __name__ == '__main__':
    rootdir = 'E:/Zhu/ETreg/newdata/HUA3/'
    dir_dir = rootdir + 'TOM/stack/ori/'
    output_dir = rootdir + '/TOM/oriava/'

    dirnamelist = os.listdir(dir_dir)
    dirnamelist.sort(key=lambda x: int(x))
    for i in range(len(dirnamelist)):
        imgdir = dir_dir + dirnamelist[i] + '/'

        imgavg = AvergeImg(imgdir)
        cv2.imwrite(output_dir + str(i + 1).zfill(2) + '.tif', imgavg)