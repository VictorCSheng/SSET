import os
import cv2
import numpy as np
import scipy.io as scio

from csutil import util
from csutil.imgfeature import ImageFeature
from csutil.coarsealign import AlignViaPro

if __name__ == '__main__':
    scale_factorreg = 4
    constant_scalfac = 4

    display_height = 1080
    display_weight = 1080  # 2160

    root_dir = 'E:/Zhu/ETreg/newdata/HUA3/'

    TEMb_ori_dir = root_dir + '/TEMb/ori/'
    TEMb_coarse_dir = root_dir + '/TEMb/coarse/'

    TEMs_ori_dir = root_dir + '/TEMs/ori/'
    TEMs_coarse_s_dir = root_dir + '/TEMs/coarse_s/'
    TEMs_coarse_b_dir = root_dir + '/TEMs/coarse_b/'

    TOM_ori_dir = root_dir + '/TOM/oriava/'
    TOM_coarse_s_dir = root_dir + '/TOM/coarse_s/'
    TOM_coarse_b_dir = root_dir + '/TOM/coarse_b/'

    TEMb_reg_config_dir = root_dir + '/reg_config/TEMb/'
    TEMs_reg_config_dir = root_dir + '/reg_config/TEMs/'
    TOM_reg_config_dir = root_dir + '/reg_config/TOM/'

    # TEMb
    util.mkdir(TEMb_coarse_dir)
    # TEMs
    util.mkdir(TEMs_coarse_s_dir)
    util.mkdir(TEMs_coarse_b_dir)
    # TOM
    util.mkdir(TOM_ori_dir)
    util.mkdir(TOM_coarse_s_dir)
    util.mkdir(TOM_coarse_b_dir)

    # reg_config
    util.mkdir(TEMb_reg_config_dir)
    util.mkdir(TEMs_reg_config_dir)
    util.mkdir(TOM_reg_config_dir)

    imgnamelist = os.listdir(TEMb_ori_dir)
    imgnamelist = sorted(imgnamelist)

    # TEMb and TEMb
    for i in range(len(imgnamelist) - 1):
        if i == 0:
            img_TEMb_1ori = cv2.imread(TEMb_ori_dir + imgnamelist[i])
        else:
            img_TEMb_1ori = cv2.imread(TEMb_coarse_dir + imgnamelist[i])
        if img_TEMb_1ori.ndim == 3:
            img_TEMb_1ori = cv2.cvtColor(img_TEMb_1ori, cv2.COLOR_RGB2GRAY)

        img_TEMb_2ori = cv2.imread(TEMb_ori_dir + imgnamelist[i + 1])
        if img_TEMb_2ori.ndim == 3:
            img_TEMb_2ori = cv2.cvtColor(img_TEMb_2ori, cv2.COLOR_RGB2GRAY)

        #
        img_TEMb_1ori = cv2.equalizeHist(img_TEMb_1ori)
        img_TEMb_2ori = cv2.equalizeHist(img_TEMb_2ori)
        #
        img_TEMb_1ori = cv2.medianBlur(img_TEMb_1ori, 11)
        img_TEMb_2ori = cv2.medianBlur(img_TEMb_2ori, 11)

        #
        imgtemp1 = cv2.resize(img_TEMb_1ori, (int(img_TEMb_1ori.shape[1] / (scale_factorreg * constant_scalfac)), int(img_TEMb_1ori.shape[0] / (scale_factorreg * constant_scalfac))),
                              interpolation=cv2.INTER_AREA)
        imgtemp2 = cv2.resize(img_TEMb_2ori, (int(img_TEMb_2ori.shape[1] / (scale_factorreg * constant_scalfac)), int(img_TEMb_2ori.shape[0] / (scale_factorreg * constant_scalfac))),
                              interpolation=cv2.INTER_AREA)

        #
        img1feature = ImageFeature(imgtemp1)
        kp1, des1 = img1feature._detect_and_compute()
        img2feature = ImageFeature(imgtemp2)
        kp2, des2 = img2feature._detect_and_compute()

        #
        Aligner = AlignViaPro(img2feature, img1feature, kp2, kp1, des2, des1, method="rigid", ransacReprojThreshold=20)
        matches, matcheswithoutlist, pts_f, pts_m = Aligner.get_match()

        mat, inters = Aligner.get_deformation()

        matlist = mat.tolist()
        matlist.append([0, 0, 1])
        mat = np.array(matlist)
        scale_TEMb = np.array(
            [[1 / (constant_scalfac * scale_factorreg), 0, 0], [0, 1 / (constant_scalfac * scale_factorreg), 0],
             [0, 0, 1]])
        mattemp = np.matmul(mat, scale_TEMb)
        mat_new = np.matmul(np.linalg.inv(scale_TEMb), mattemp)

        # save
        scio.savemat(TEMb_reg_config_dir + 'rigid_' + str(i).zfill(2) + '_' + str(i+1).zfill(2) + '.mat', {'mat': mat_new})

        img2_align = cv2.warpPerspective(img_TEMb_2ori, mat_new, (img_TEMb_1ori.shape[1], img_TEMb_1ori.shape[0]))
        cv2.imwrite(TEMb_coarse_dir + imgnamelist[i + 1], img2_align)
        if i == 0:
            cv2.imwrite(TEMb_coarse_dir + imgnamelist[i], img_TEMb_1ori)

    #
    warp_TEMstob_matlist = []
    for i in range(len(imgnamelist)):
        img_TEMb = cv2.imread(TEMb_coarse_dir + imgnamelist[i])
        if img_TEMb.ndim == 3:
            img_TEMb = cv2.cvtColor(img_TEMb, cv2.COLOR_RGB2GRAY)

        img_TEMs = cv2.imread(TEMs_ori_dir + imgnamelist[i])
        if img_TEMs.ndim == 3:
            img_TEMs = cv2.cvtColor(img_TEMs, cv2.COLOR_RGB2GRAY)

        #
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(
                                8, 8))  # clipLimit颜色对比度的阈值， titleGridSize进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
        img_TEMb = clahe.apply(img_TEMb)
        img_TEMs = clahe.apply(img_TEMs)

        img_TEMb = cv2.medianBlur(img_TEMb, 11)
        img_TEMs = cv2.medianBlur(img_TEMs, 11)

        img1_resize = cv2.resize(img_TEMb, (
        int(img_TEMb.shape[1] / constant_scalfac), int(img_TEMb.shape[0] / constant_scalfac)),
                                     interpolation=cv2.INTER_CUBIC)
        img2_resize = cv2.resize(img_TEMs, (int(img_TEMs.shape[1] / (constant_scalfac * scale_factorreg)),
                                                int(img_TEMs.shape[0] / (constant_scalfac * scale_factorreg))),
                                     interpolation=cv2.INTER_CUBIC)

        img1feature = ImageFeature(img1_resize)
        kp1, des1 = img1feature._detect_and_compute()
        img2feature = ImageFeature(img2_resize)
        kp2, des2 = img2feature._detect_and_compute()

        #
        Aligner = AlignViaPro(img2_resize, img1_resize, kp2, kp1, des2, des1, method="rigid",
                              ransacReprojThreshold=20)
        matches, matcheswithoutlist, pts_f, pts_m = Aligner.get_match()

        mat, inters = Aligner.get_deformation()

        matlist = mat.tolist()
        matlist.append([0, 0, 1])
        mat = np.array(matlist)
        scale_TEMstoTEMb = np.array(
            [[1 / (constant_scalfac * scale_factorreg), 0, 0], [0, 1 / (constant_scalfac * scale_factorreg), 0],
             [0, 0, 1]])
        mattemp = np.matmul(mat, scale_TEMstoTEMb)
        scale_TEM = np.array([[constant_scalfac, 0, 0], [0, constant_scalfac, 0], [0, 0, 1]])
        mat_s = np.matmul(scale_TEM, mattemp)

        warp_TEMstob_matlist.append((mat_s))

        img2_align_s = cv2.warpPerspective(img_TEMs, mat_s, (img_TEMb.shape[1], img_TEMb.shape[0]))

        # save
        cv2.imwrite(TEMs_coarse_s_dir + imgnamelist[i], img2_align_s)
        scio.savemat(TEMs_reg_config_dir + 'rigid_s' + str(i).zfill(2) + '.mat',
                     {'mat': mat_s})

    #
    warp_TOMtoTEMs_matlist = []
    for i in range(len(imgnamelist)):
        img_TEMs = cv2.imread(TEMs_ori_dir + imgnamelist[i])
        if img_TEMs.ndim == 3:
            img_TEMs = cv2.cvtColor(img_TEMs, cv2.COLOR_RGB2GRAY)

        img_TOM = cv2.imread(TOM_ori_dir + imgnamelist[i])
        if img_TOM.ndim == 3:
            img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
        img_TOM = util.imgRotate(img_TOM, 90)

        #
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))  # clipLimit颜色对比度的阈值， titleGridSize进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
        img_TEMs = clahe.apply(img_TEMs)
        img_TOM = clahe.apply(img_TOM)

        img_TEMs = cv2.medianBlur(img_TEMs, 11)
        img_TOM = cv2.medianBlur(img_TOM, 11)

        img_TEMs_resize = cv2.resize(img_TEMs, (int(img_TEMs.shape[1] / (constant_scalfac * scale_factorreg)),
                                                int(img_TEMs.shape[0] / (constant_scalfac * scale_factorreg))),
                                     interpolation=cv2.INTER_CUBIC)
        img_TOM_resize = cv2.resize(img_TOM, (int(img_TOM.shape[1] / (constant_scalfac * scale_factorreg)),
                                              int(img_TOM.shape[0] / (constant_scalfac * scale_factorreg))),
                                    interpolation=cv2.INTER_CUBIC)

        img1feature = ImageFeature(img_TEMs_resize)
        kp1, des1 = img1feature._detect_and_compute()
        img2feature = ImageFeature(img_TOM_resize)
        kp2, des2 = img2feature._detect_and_compute()

        #
        Aligner = AlignViaPro(img_TOM_resize, img_TEMs_resize, kp2, kp1, des2, des1, method="rigid",
                              ransacReprojThreshold=3)
        matches, matcheswithoutlist, pts_f, pts_m = Aligner.get_match()

        mat, inters = Aligner.get_deformation()

        matlist = mat.tolist()
        matlist.append([0, 0, 1])
        mat = np.array(matlist)
        scale_s = np.array(
            [[1 / (constant_scalfac * scale_factorreg), 0, 0], [0, 1 / (constant_scalfac * scale_factorreg), 0],
             [0, 0, 1]])
        mattemp = np.matmul(mat, scale_s)
        scale_s_inverse = np.array([[(constant_scalfac * scale_factorreg), 0, 0], [0, (constant_scalfac * scale_factorreg), 0], [0, 0, 1]])
        mat_temp = np.matmul(scale_s_inverse, mattemp)
        mat_new = np.matmul(warp_TEMstob_matlist[i], mat_temp)

        warp_TOMtoTEMs_matlist.append((mat_new))
        img2_align_s = cv2.warpPerspective(img_TOM, mat_new, (img_TEMs.shape[1], img_TEMs.shape[0]))

        #
        cv2.imwrite(TOM_coarse_s_dir + imgnamelist[i], img2_align_s)
        scio.savemat(TOM_reg_config_dir + 'rigid_s' + str(i).zfill(2) + '.mat',
                     {'mat': mat_new})

    #
    #
    deformation_mat, new_x_length, new_y_length = util.imgintersection(TEMs_coarse_s_dir, TEMs_reg_config_dir, imgnamelist)
    scio.savemat(TEMs_reg_config_dir + 'areaselct' + '.mat',
                 {'mat': deformation_mat})
    scio.savemat(TEMs_reg_config_dir + 'new_x_length' + '.mat',
                 {'mat': new_x_length})
    scio.savemat(TEMs_reg_config_dir + 'new_y_length' + '.mat',
                 {'mat': new_y_length})
    #
    for i in range(len(imgnamelist)):
        img_TEMs = cv2.imread(TEMs_ori_dir + imgnamelist[i])
        if img_TEMs.ndim == 3:
            img_TEMs = cv2.cvtColor(img_TEMs, cv2.COLOR_RGB2GRAY)

        mattems = warp_TEMstob_matlist[i]
        mattems = np.matmul(deformation_mat, mattems)
        img2_align_tems = cv2.warpPerspective(img_TEMs, mattems, (img_TEMs.shape[1], img_TEMs.shape[0]))
        img2_align_temsF = img2_align_tems[0:new_y_length, 0:new_x_length]
        cv2.imwrite(TEMs_coarse_b_dir + imgnamelist[i], img2_align_temsF)

        img_TOM = cv2.imread(TOM_ori_dir + imgnamelist[i])
        if img_TOM.ndim == 3:
            img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
        img_TOM = util.imgRotate(img_TOM, 90)

        mattom = warp_TOMtoTEMs_matlist[i]
        mattom = np.matmul(deformation_mat, mattom)
        img2_align_tom = cv2.warpPerspective(img_TOM, mattom, (img_TOM.shape[1], img_TOM.shape[0]))
        img2_align_tomF = img2_align_tom[0:new_y_length, 0:new_x_length]
        cv2.imwrite(TOM_coarse_b_dir + imgnamelist[i], img2_align_tomF)


