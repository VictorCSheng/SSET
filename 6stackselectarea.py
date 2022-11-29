import os
import cv2
from csutil import util

if __name__ == '__main__':
    root_dir = 'E:/Zhu/ETreg/newdata/HUA3/'

    TOM_fine_dir = root_dir + '/TOM/fine/'
    stackdeformationdir = root_dir + '/TOM/stack/deformation/'

    resultdir = root_dir + '/TOM/stack/result/'

    util.mkdir(resultdir)

    tl_cut, br_cut = util.stackareaselect(TOM_fine_dir)

    dirnamelist = os.listdir(stackdeformationdir)
    dirnamelist = sorted(dirnamelist, key=lambda x: int(x))

    for i in range(len(dirnamelist)):
        imgdir = dirnamelist[i]
        imglist = os.listdir(stackdeformationdir + imgdir + '/')

        util.mkdir(resultdir + imgdir + '/')

        for j in range(len(imglist)):
            img_TOM = cv2.imread(stackdeformationdir + imgdir + '/' + imglist[j])
            if img_TOM.ndim == 3:
                img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
            img_TOM = img_TOM[tl_cut[1]:br_cut[1],tl_cut[0]:br_cut[0]]
            cv2.imwrite(resultdir + imgdir + '/' + imglist[j], img_TOM)