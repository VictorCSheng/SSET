# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:14:22 2020

@author: 78257
"""

import os
import numpy as np
import cv2
from PIL import Image
import scipy.io as scio
import scipy.signal as signal
import math

display_height = 1080
display_weight = 1080    #2160

#
def ResizeWithAspectRatio(image, width=None, height=1080, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        img_new_size = (math.ceil(w * r), height)
    elif height is None:
        r = width / float(w)
        img_new_size = (width, math.ceil(h * r))
    else:
        img_new_size = (width, height)

    return cv2.resize(image, img_new_size, interpolation=inter)

#
def GetoricutSize(image, display_height, tl, br):
    (h, w) = image.shape[:2]
    tl_cut = [math.ceil(tl[0] * h / display_height), math.ceil(tl[1] * h / display_height)]
    br_cut = [math.ceil(br[0] * h / display_height), math.ceil(br[1] * h / display_height)]
    return (tl_cut, br_cut)

#
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None, 'left_button_down': False}

    cv2.namedWindow(title)

    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            param['tl'] = param['current_pos']
            param['left_button_down'] = True
        if param['left_button_down'] and event == cv2.EVENT_MOUSEMOVE:
            param['br'] = None
        if param['left_button_down'] and event == cv2.EVENT_LBUTTONUP:
            param['br'] = param['current_pos']
            param['left_button_down'] = False

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (0, 0, 255))

        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(5)

    cv2.imshow(title, im_draw)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)  #tl=(y1,x1), br=(y2,x2)

#
def imgintersection(TEMs_coarse_s_dir, TEMs_reg_config_dir, imgnamelist):
    imgsum = 0
    for i in range(len(imgnamelist)):
        img_cal = 0

        img_TEMs = cv2.imread(TEMs_coarse_s_dir + imgnamelist[i])
        if img_TEMs.ndim == 3:
            img_TEMs = cv2.cvtColor(img_TEMs, cv2.COLOR_RGB2GRAY)
        img_TEMs = 255 - img_TEMs

        img_cal = img_cal + 1

        #
        erode_kernel = np.ones((10, 10), np.uint8)

        ret_tems, th_tems = cv2.threshold(img_TEMs, 250, 255, cv2.THRESH_BINARY)
        img_TEMs_th_erode = cv2.erode(th_tems, erode_kernel)

        imgsum = (img_TEMs_th_erode.astype(float) + imgsum) / img_cal
        imgsum = imgsum.astype(np.uint8)
        ret_sum, imgsum = cv2.threshold(imgsum, 5, 255, cv2.THRESH_BINARY)
        imgsum = cv2.erode(imgsum, erode_kernel)

    img_temp = cv2.imread(TEMs_coarse_s_dir + imgnamelist[int(len(imgnamelist) / 2)])
    if img_temp.ndim == 3:
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
    img_temp = imgsum.astype(float) + img_temp.astype(float)
    img_temp[img_temp >= 255] = 255
    img_temp = img_temp.astype(np.uint8)

    imgresize = ResizeWithAspectRatio(img_temp, width=None, height=display_height)
    (tl, br) = get_rect(imgresize, title='get_rect')

    (tl_cut, br_cut) = GetoricutSize(img_temp, display_height, tl, br)
    print(tl_cut, br_cut)

    cv2.waitKey()
    cv2.destroyAllWindows()

    lefttopx = tl_cut[0]
    lefttopy = tl_cut[1]
    rightbottomx = br_cut[0]
    rightbottomy = br_cut[1]

    data = scio.loadmat(TEMs_reg_config_dir + 'rigid_s' + str(int(len(imgnamelist) / 2)).zfill(2) + '.mat')
    mat_TEMs2TEMbcoarse = data['mat']
    lefttoppoint = np.array([lefttopx, lefttopy, 1])
    lefttoppointori = np.matmul(np.linalg.inv(mat_TEMs2TEMbcoarse), lefttoppoint)
    rightbottompoint = np.array([rightbottomx, rightbottomy, 1])
    rightbottompointori = np.matmul(np.linalg.inv(mat_TEMs2TEMbcoarse), rightbottompoint)

    movemat = np.array([[1, 0, -lefttopx], [0, 1, -lefttopy], [0, 0, 1]])
    scalenum = min(np.floor((rightbottompointori[0] - lefttoppointori[0]) / (rightbottomx - lefttopx)),
                   np.floor((rightbottompointori[1] - lefttoppointori[1]) / (rightbottomy - lefttopy)))
    scalmat = np.array([[scalenum, 0, 0], [0, scalenum, 0], [0, 0, 1]])
    new_x_length = int(scalenum * (rightbottomx - lefttopx))
    new_y_length = int(scalenum * (rightbottomy - lefttopy))
    deformation_mat = np.matmul(scalmat, movemat)

    return deformation_mat, new_x_length, new_y_length

#
def stackareaselect(TOM_fine_dir):
    imgnamelist = os.listdir(TOM_fine_dir)
    imgsum = 0
    for i in range(len(imgnamelist)):
        img_cal = 0

        img_TOM = cv2.imread(TOM_fine_dir + imgnamelist[i])
        if img_TOM.ndim == 3:
            img_TOM = cv2.cvtColor(img_TOM, cv2.COLOR_RGB2GRAY)
        img_TOM = 255 - img_TOM

        img_cal = img_cal + 1

        #
        erode_kernel = np.ones((10, 10), np.uint8)

        ret_tems, th_tems = cv2.threshold(img_TOM, 250, 255, cv2.THRESH_BINARY)
        img_TOM_th_erode = cv2.erode(th_tems, erode_kernel)

        imgsum = (img_TOM_th_erode.astype(float) + imgsum) / img_cal
        imgsum = imgsum.astype(np.uint8)
        ret_sum, imgsum = cv2.threshold(imgsum, 5, 255, cv2.THRESH_BINARY)
        imgsum = cv2.erode(imgsum, erode_kernel)

    img_temp = cv2.imread(TOM_fine_dir + imgnamelist[int(len(imgnamelist) / 2)])
    if img_temp.ndim == 3:
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
    img_temp = imgsum.astype(float) + img_temp.astype(float)
    img_temp[img_temp >= 255] = 255
    img_temp = img_temp.astype(np.uint8)

    imgresize = ResizeWithAspectRatio(img_temp, height=display_height)
    (tl, br) = get_rect(imgresize, title='get_rect')

    (tl_cut, br_cut) = GetoricutSize(img_temp, display_height, tl, br)
    print(tl_cut, br_cut)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return tl_cut, br_cut


#
def load_Img(imgDir,imgFoldName):
     imgs = os.listdir(imgDir+imgFoldName)
     imgNum = len(imgs)
     data = np.empty((imgNum,1,12,12),dtype="float32")
     label = np.empty((imgNum,),dtype="uint8")
     for i in range (imgNum):
         img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
         arr = np.asarray(img,dtype="float32")
         data[i,:,:,:] = arr
         label[i] = int(imgs[i].split('.')[0])
     return data,label

#
def transfer_16bit_to_8bit(image_16bit):
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    #
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

#
import matplotlib.pyplot as plt
def mrcimgshow(mrcdata):
    #
    plt.ion()
    fig = plt.figure('MRC')
    # fig2 = plt.figure('subImg')
    
    for i in range(0, len(mrcdata)):
    	#
        img = mrcdata[i,:,:]
        img = transfer_16bit_to_8bit(img)

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.axis('off')  #
        ax1.imshow(img, cmap='gray')
        plt.pause(0.2)
        fig.clf()
    
    plt.ioff()

def listimgshow(listimg):
    plt.ion()
    fig = plt.figure('LIST')
    # fig2 = plt.figure('subImg')
    
    for i in range(0, len(listimg)):
    	#
        img = listimg[i]

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.axis('off')  # 关掉坐标轴
        ax1.imshow(img, cmap='gray')
    #	ax1.plot(p1[:, 0], p1[:, 1], 'g.')
        plt.pause(0.2)
        fig.clf()
    
    plt.ioff()

#
import imageio
def create_gif(image_list, gif_name, duration = 0.3):
    frames = []
    for i in range(len(image_list)):
    	#
#        img = image_list[i,:,:]
        img = image_list[i]
        img = transfer_16bit_to_8bit(img)
        frames.append(img)

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

#
def kp_to_numpy(kplist):
    kp_lst = []
    for kp in kplist:
        kp_lst.append(kp.pt)
    kp_lst = np.array(kp_lst)
    return kp_lst


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    #
    if not isExists:
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

def get_filename(path,filetype):
    name =[]
    for root,dirs,files in os.walk(path):
        for i in files:
            if filetype in i:
                name.append(i)
    return name

def unionarea(imglist):
    imgtemp = np.zeros(imglist[0].shape)
    imgtemp = 255 * imgtemp

    imglefttopx_list = []
    imglefttopy_list = []
    imgrightbottomx_list = []
    imgrightbottomy_list = []
    for i in range(len(imglist)):
        img = imglist[i]
        imgid = np.where(img != 0)
        imgidx = imgid[0]
        imgidy = imgid[1]
        imglefttopx = imgidx.min()
        imglefttopx_list.append(imglefttopx)
        imglefttopy = imgidy.min()
        imglefttopy_list.append(imglefttopy)
        imgrightbottomx = imgidx.max()
        imgrightbottomx_list.append(imgrightbottomx)
        imgrightbottomy = imgidy.max()
        imgrightbottomy_list.append(imgrightbottomy)

    lefttopx = min(imglefttopx_list)
    lefttopy = min(imglefttopy_list)
    rightbottomx = max(imgrightbottomx_list)
    rightbottomy = max(imgrightbottomy_list)

    imgmask = np.ones((rightbottomx - lefttopx, rightbottomy - lefttopy)) * 255
    imgtemp[lefttopx:rightbottomx, lefttopy:rightbottomy] = imgmask
    return imgtemp, (float(lefttopx), float(lefttopy)), (float(rightbottomx), float(rightbottomy))

#
def unionareamin(img1,img2):
    imglefttopy_list1 = []
    imgrightbottomy_list1 = []

    imglefttopy_list2 = []
    imgrightbottomy_list2 = []

    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    for i in range(img1.shape[0]):
        img1idrow = np.where(img1[i,:] != 0)
        img1idyrow = img1idrow[0]

        if img1idyrow.size == 0:
            continue

        imglefttopy_list1.append(img1idyrow.min())
        imgrightbottomy_list1.append(img1idyrow.max())

        img2idrow = np.where(img2[i,:] != 0)
        img2idyrow = img2idrow[0]

        if img2idyrow.size == 0:
            continue

        imglefttopy_list2.append(img2idyrow.min())
        imgrightbottomy_list2.append(img2idyrow.max())

    imglefttopy_list1 = signal.medfilt(imglefttopy_list1, 5)
    imglefttopy_list2 = signal.medfilt(imglefttopy_list2, 5)
    # imglefttopy_np = np.array(imglefttopy_list1)
    # imglefttopy_np[imglefttopy_np == 255] = x
    # img1idrow = np.where(imglefttopy_np == (img1.shape[0]-1))
    lefttopy1 = max(imglefttopy_list1)
    lefttopy2 = max(imglefttopy_list2)
    if lefttopy1 > lefttopy2:
        lefttopy = lefttopy1
        imglefttopx_list = np.where(imglefttopy_list1 == lefttopy1)
        lefttopx = min(imglefttopx_list[0])
    else:
        lefttopy = lefttopy2
        imglefttopx_list = np.where(imglefttopy_list2 == lefttopy2)
        lefttopx = min(imglefttopx_list[0])

    imgrightbottomy_list1 = signal.medfilt(imgrightbottomy_list1, 5)
    imgrightbottomy_list2 = signal.medfilt(imgrightbottomy_list2, 5)
    rightbottomy1 = min(imgrightbottomy_list1)
    rightbottomy2 = min(imgrightbottomy_list2)
    if rightbottomy1 < rightbottomy2:
        rightbottomy = rightbottomy1
        imgrightbottomx_list = np.where(imgrightbottomy_list1 == rightbottomy1)
        rightbottomx = max(imgrightbottomx_list[0])
    else:
        rightbottomy = rightbottomy2
        imgrightbottomx_list = np.where(imgrightbottomy_list2 == rightbottomy2)
        rightbottomx = max(imgrightbottomx_list[0])


    imgmask = np.zeros(img1.shape[0:2])
    imgmask[lefttopx:rightbottomx, lefttopy:rightbottomy] = np.ones((rightbottomx - lefttopx, rightbottomy - lefttopy)) * 255

    return imgmask, float(lefttopx), float(lefttopy), float(rightbottomx), float(rightbottomy)

# def unionareamin_point(img1point,img2,mat):
#     xmin = 0
#     ymin = 0
#
#     xmax = img2.shape.shape[1]
#     ymax = img2.shape.shape[0]
#
#     lefttop = np.array([xmin, ymin, 1])
#     leftbottom = np.array([xmin, ymax, 1])
#
#     righttop = np.array([xmax, ymin, 1])
#     rightbottom = np.array([xmax, ymax, 1])
#
#     temp = np.round(mat.dot(lefttop))
#     if temp[0] >= 0 and temp[1] >= 0:
#         lefttopx = temp[0]
#         lefttopy = temp[1]
#     elif temp[0] >= 0:
#         lefttopx = temp[0]
#         lefttopy = temp[1]

def read16bitimage(img_path):
    uint16_img = cv2.imread(img_path, -1)
    uint16_img -= uint16_img.min()
    uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
    uint16_img *= 255
    uint8_img = uint16_img.astype(np.uint8)
    return uint8_img

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

def MaxImg(imgdir):
    imgnamelist = os.listdir(imgdir)
    img_max = cv2.imread(imgdir + imgnamelist[0])
    if img_max.ndim == 3:
        img_max = cv2.cvtColor(img_max, cv2.COLOR_RGB2GRAY)
    img_buffer = np.zeros((2, img_max.shape[1], img_max.shape[0]))
    img_buffer[0,:,:] = img_max
    for i in range(len(imgnamelist-1)):
        img_ori = cv2.imread(imgdir + imgnamelist[i+1])
        if img_ori.ndim == 3:
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
        img_buffer[1, :, :] = img_ori
        img_max = np.amax(img_buffer, axis=0)
        img_buffer[0,:,:] = img_max

    return img_max.astype(np.uint8)

def imgRotate(img, angle):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = img.shape
    center_x = w // 2
    center_y = h // 2
    center = (center_x, center_y)

    M = cv2.getRotationMatrix2D(center, angle, 1.)
    img = cv2.warpAffine(img, M, (w, h))
    return img