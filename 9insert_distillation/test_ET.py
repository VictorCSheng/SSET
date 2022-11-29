import os
import cv2
import torch
from tifffile import imread, imsave
import numpy as np
import ET_model
from torchvision import transforms

def interpolate_image(frames, imindex, factor, flow, interp, back_warp, file_out, orsize, device):   # frames是要处理的图像， factor缩放因子似乎是要插图的数目+1
    frame0 = torch.stack(frames[:-1])  #
    frame1 = torch.stack(frames[1:])  #

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)  #

    flow_out = flow(ix)  #
    f01 = flow_out[:, :2, :, :]  #
    f10 = flow_out[:, 2:, :, :]  #

    for i in range(1, factor):
        t = i / factor   #
        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10     #
        ft1 = co_eff[2] * f01 + co_eff[3] * f10     #

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)    #

        ft0f = io[:, :2, :, :] + ft0  #
        ft1f = io[:, 2:4, :, :] + ft1 #
        vt0 = torch.sigmoid(io[:, 4:5, :, :])  #
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)   #
        gi1ft1f = back_warp(i1, ft1f)   #

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)  #

        imgtemp = ft_p.cpu()[0]  #
        imgtemp = imgtemp.permute([1, 2, 0])
        imgtemp = imgtemp.detach().numpy()
        # imgtemp = imgtemp.swapaxes(0, 2)
        imgtemp = imgtemp * 255
        imgtemp = np.around(imgtemp)
        imgtemp = np.array(imgtemp, dtype=np.uint8)
        imgtemp = cv2.resize(imgtemp, (orsize[1], orsize[0]), interpolation=cv2.INTER_CUBIC)
        imsave(file_out + str(imindex[0] + i).zfill(3) + ".tif", imgtemp)


def generate_img(file_in, file_out, orsize, factor, flow, interp, back_warp, device):
    trans_forward = transforms.ToTensor()

    frames = []
    imindex = []

    imgnamelist = os.listdir(file_in)
    imgnamelist = sorted(imgnamelist)

    # im_frameF = imread(os.path.join(file_in, imgnamelist[0]))
    im_frameF = cv2.imread(os.path.join(file_in, imgnamelist[0]))
    im_frameF = im_frameF / 255
    im_frameF = im_frameF.astype(np.float32)
    im_frameF = cv2.resize(im_frameF, (w, h), interpolation=cv2.INTER_AREA)
    im_frameF = trans_forward(im_frameF)
    frames.append(im_frameF)
    imgindextemp = int(imgnamelist[0][0:-4])
    imindex.append(imgindextemp)

    # im_frameL = imread(os.path.join(file_in, imgnamelist[1]))
    im_frameL = cv2.imread(os.path.join(file_in, imgnamelist[1]))
    im_frameL = im_frameL / 255
    im_frameL = im_frameL.astype(np.float32)
    im_frameL = cv2.resize(im_frameL, (w, h), interpolation=cv2.INTER_AREA)
    im_frameL = trans_forward(im_frameL)
    frames.append(im_frameL)
    imgindextemp = int(imgnamelist[1][0:-4])
    imindex.append(imgindextemp)

    interpolate_image(frames, imindex, factor, flow, interp, back_warp, file_out, orsize, device)
    # intermediate_frames = list(zip(*intermediate_frames))  # *解[](括号)  zip将所有东西压到一个元组里
    #
    # imgid = 1
    # for fid, iframe in enumerate(intermediate_frames):
    #     for frm in iframe:
    #         imgtemp = frm.cpu()[0]  # 提取numpy
    #         imgtemp = imgtemp * 255
    #         imgtemp = np.around(imgtemp)
    #         imgtemp = np.array(imgtemp, dtype=np.uint8)
    #         imgtemp = cv2.resize(imgtemp, (w0, h0), interpolation=cv2.INTER_CUBIC)
    #         imsave(file_out + str(imindex[0] + imgid).zfill(3) + ".tif", imgtemp)
    #         imgid = imgid + 1

if __name__ == "__main__":
    #
    np.random.seed(0)
    torch.manual_seed(0)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    checkpoint_dir = "./save/"

    file_in = "./data/result/frames/"
    file_out = "./data/result/insert/"
    factor = 57         # stack1 63+1 124+1  stack4 88+1   1207: 45    48    37    51    68    52    52 ribbon:19    30    62    26    55    25    26    45


    imgnamelist = os.listdir(file_in)
    # im_frameF = imread(os.path.join(file_in, imgnamelist[0]))
    im_frameF = cv2.imread(os.path.join(file_in, imgnamelist[0]))
    if len(im_frameF.shape) == 3:
        w0, h0, c0 = im_frameF.shape
    else:
        w0, h0 = im_frameF.shape
    orsize = [w0, h0]
    w, h = (w0 // 32) * 32, (h0 // 32) * 32  # setup_back_warp（网络）要是32的整数倍数？？？？？？？？

    with torch.set_grad_enabled(False):
        flow = ET_model.ET_model(6, 4).to(device)
        interp = ET_model.ET_model(20, 5).to(device)
        back_warp = ET_model.backWarp(w, h, device).to(device)

    ##
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    states = torch.load(checkpoint_dir + "/student_fineturne/SuperSloMo_fine_003.ckpt")  # synapis 3  ribbon 1
    # states = torch.load(checkpoint, map_location='cpu')  #
    flow.load_state_dict(states['state_dictFC'])
    interp.load_state_dict(states['state_dictAT'])

    generate_img(file_in, file_out, orsize, factor, flow, interp, back_warp, device)

