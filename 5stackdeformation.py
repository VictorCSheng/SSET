import os
import cv2
import numpy as np
import scipy.io as scio
from csutil import util


if __name__ == '__main__':
    root_dir = 'E:/Zhu/ETreg/newdata/HUA3/'

    stackdir = root_dir + '/TOM/stack/ori/'
    TEMs_reg_config_dir = root_dir + '/reg_config/TEMs/'
    TOM_reg_config_dir = root_dir + '/reg_config/TOM/'
    resultdir = root_dir + '/TOM/stack/deformation/'

    util.mkdir(resultdir)

    dirnamelist = os.listdir(stackdir)
    dirnamelist = sorted(dirnamelist, key=lambda x: int(x))

    data = scio.loadmat(TEMs_reg_config_dir + 'areaselct' + '.mat')
    deformation_mat = data['mat']
    data = scio.loadmat(TEMs_reg_config_dir + 'new_x_length' + '.mat')
    new_x_length = data['mat']
    new_x_length = new_x_length[0][0]
    data = scio.loadmat(TEMs_reg_config_dir + 'new_y_length' + '.mat')
    new_y_length = data['mat']
    new_y_length = new_y_length[0][0]

    for i in range(len(dirnamelist)):
        imgdir = dirnamelist[i]
        imglist = os.listdir(stackdir + imgdir + '/')

        util.mkdir(resultdir + imgdir + '/')

        data = scio.loadmat(TOM_reg_config_dir + 'rigid_s' + str(i).zfill(2) + '.mat')
        mattom = data['mat']
        mattom = np.matmul(deformation_mat, mattom)
        if i == 0:
            for j in range(len(imglist)):
                img_TOM = cv2.imread(stackdir + imgdir + '/' + imglist[j])
                if img_TOM.ndim == 3:
                    img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
                img_TOM = util.imgRotate(img_TOM, 90)
                imgTOM_align = cv2.warpPerspective(img_TOM, mattom, (img_TOM.shape[1], img_TOM.shape[0]))
                imgTOM_align = imgTOM_align[0:new_y_length, 0:new_x_length]
                cv2.imwrite(resultdir + imgdir + '/' + imglist[j], imgTOM_align)
        else:
            data = scio.loadmat(TEMs_reg_config_dir + 'fine_rigid_' + str(i - 1).zfill(2) + '_' + str(i).zfill(2) + '.mat')
            finetmpmat = data['mat']
            mattom_fine = np.matmul(finetmpmat, mattom)

            data = scio.loadmat(TEMs_reg_config_dir + 'TMlocc_grid' + str(i-1) + '.mat')
            TMlocc_grid = data['TMlocc_grid']
            data = scio.loadmat(TEMs_reg_config_dir + 'final_loc' + str(i-1) + '.mat')
            final_loc = data['final_loc']
            matches = []
            for n in range(1, TMlocc_grid.shape[1] + 1):
                matches.append(cv2.DMatch(n, n, 0))

            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.setRegularizationParameter(100000)
            tps.estimateTransformation(final_loc, TMlocc_grid, matches)  # 接口（模板特征点，目标特征点，匹配点）
            for j in range(len(imglist)):
                img_TOM = cv2.imread(stackdir + imgdir + '/' + imglist[j])
                if img_TOM.ndim == 3:
                    img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
                img_TOM = util.imgRotate(img_TOM, 90)
                imgTOM_align = cv2.warpPerspective(img_TOM, mattom_fine, (img_TOM.shape[1], img_TOM.shape[0]))
                imgTOM_align = imgTOM_align[0:new_y_length, 0:new_x_length]
                img = tps.warpImage(imgTOM_align)
                cv2.imwrite(resultdir + imgdir + '/' + imglist[j], img)