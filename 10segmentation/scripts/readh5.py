import h5py
import cv2
import numpy as np
path = 'E:/Zhu/ETreg/code/segmentation/pytorch_connectomics/dataset/output/result.h5'
outputdir = 'E:/Zhu/ETreg/code/segmentation/pytorch_connectomics/dataset/output/img/'
dataset = h5py.File(path, 'r')
for key in dataset.keys():
    imgdatasets = np.array(dataset[key])
    for j in range(imgdatasets.shape[1]):
        imgtemp = imgdatasets[0, j, :, :]
        cv2.imwrite(outputdir + str(j) + '.tif', imgtemp)
