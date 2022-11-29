import os
import cv2
import numpy as np
import scipy.io as scio

from csutil import util
from csutil.imgfeature import ImageFeature
from csutil.coarsealign import AlignViaPro

if __name__ == '__main__':
    #
    scale_factorreg = 12

    root_dir = 'E:/Zhu/ETreg/newdata/HUA3/'

    TEMs_coarse_b_dir = root_dir + '/TEMs/coarse_b/'
    TEMs_finetemp_dir = root_dir + '/TEMs/finetemp/'
    TOM_coarse_b_dir = root_dir + '/TOM/coarse_b/'
    TOM_finetemp_dir = root_dir + '/TOM/finetemp/'

    TEMs_reg_config_dir = root_dir + '/reg_config/TEMs/'
    TOM_reg_config_dir = root_dir + '/reg_config/TOM/'

    # TEMs
    util.mkdir(TEMs_finetemp_dir)
    # TOM
    util.mkdir(TOM_finetemp_dir)

    imgnamelist = os.listdir(TEMs_coarse_b_dir)
    imgnamelist = sorted(imgnamelist)

    # TEMs and TEMs
    matTemsfine_list = []
    for i in range(len(imgnamelist) - 1):
        if i == 0:
            img_TEMs_1ori = cv2.imread(TEMs_coarse_b_dir + imgnamelist[i])
            cv2.imwrite(TEMs_finetemp_dir + imgnamelist[i], img_TEMs_1ori)
            imgheight = img_TEMs_1ori.shape[1]
            imgwidth = img_TEMs_1ori.shape[0]

            imgheight_temp_half = int(imgheight / 5 * 2)   ## hua3 之前都为1/3
            imgwidth_temp_half = int(imgwidth / 5 * 2)
            imgheight_tempst = int(imgheight/2) - imgheight_temp_half
            imgwidth_tempst = int(imgwidth / 2) - imgwidth_temp_half
        else:
            img_TEMs_1ori = cv2.imread(TEMs_finetemp_dir + imgnamelist[i])
        if img_TEMs_1ori.ndim == 3:
            img_TEMs_1ori = cv2.cvtColor(img_TEMs_1ori, cv2.COLOR_RGB2GRAY)

        img_TEMs_2ori = cv2.imread(TEMs_coarse_b_dir + imgnamelist[i + 1])
        if img_TEMs_2ori.ndim == 3:
            img_TEMs_2ori = cv2.cvtColor(img_TEMs_2ori, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(
                                    8, 8))  # clipLimit颜色对比度的阈值， titleGridSize进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
        img_TEMs_1ori = clahe.apply(img_TEMs_1ori)
        img_TEMs_2ori = clahe.apply(img_TEMs_2ori)

        img_TEMs_1ori = cv2.medianBlur(img_TEMs_1ori, 11)
        img_TEMs_2ori = cv2.medianBlur(img_TEMs_2ori, 11)

        img_TEMs_2ori_temp = img_TEMs_2ori[imgwidth_tempst : imgwidth_tempst + imgwidth_temp_half * 2,
                             imgheight_tempst : imgheight_tempst + imgheight_temp_half * 2]

        results = cv2.matchTemplate(img_TEMs_1ori, img_TEMs_2ori_temp,
                                    cv2.TM_CCOEFF_NORMED)  # cv2.TM_SQDIFF_NORMED利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results)  # max_loc
        offset_x = max_loc[0] - imgheight_tempst
        offset_y = max_loc[1] - imgwidth_tempst

        movemat = np.array([[1, 0., offset_x], [0., 1, offset_y], [0., 0., 1]])
        matTemsfine_list.append(movemat)
        img2_align = cv2.warpPerspective(img_TEMs_2ori, movemat, (img_TEMs_1ori.shape[1], img_TEMs_1ori.shape[0]))
        cv2.imwrite(TEMs_finetemp_dir + imgnamelist[i + 1], img2_align)
        # save
        scio.savemat(TEMs_reg_config_dir + 'fine_rigid_' + str(i).zfill(2) + '_' + str(i + 1).zfill(2) + '.mat',
                     {'mat': movemat})

        ##
        if i == 0:
            img_TOM = cv2.imread(TOM_coarse_b_dir + imgnamelist[i])
            cv2.imwrite(TOM_finetemp_dir + imgnamelist[i], img_TOM)
            img_TOM2 = cv2.imread(TOM_coarse_b_dir + imgnamelist[i + 1])
            if img_TOM2.ndim == 3:
                img_TOM2 = cv2.cvtColor(img_TOM2, cv2.COLOR_RGB2GRAY)
            img_TOM2_align = cv2.warpPerspective(img_TOM2, movemat, (img_TEMs_1ori.shape[1], img_TEMs_1ori.shape[0]))
            cv2.imwrite(TOM_finetemp_dir + imgnamelist[i + 1], img_TOM2_align)
        else:
            img_TOM = cv2.imread(TOM_coarse_b_dir + imgnamelist[i + 1])
            if img_TOM.ndim == 3:
                img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
            img_TOM_align = cv2.warpPerspective(img_TOM, movemat, (img_TEMs_1ori.shape[1], img_TEMs_1ori.shape[0]))
            cv2.imwrite(TOM_finetemp_dir + imgnamelist[i + 1], img_TOM_align)


