import os
import cv2
import torch
from PIL import Image
import numpy as np
import model
from torchvision import transforms
from torch.functional import F

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()
if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])

flow = model.UNet(6, 4).to(device)
interp = model.UNet(20, 5).to(device)
back_warp = None

def setup_back_warp(w, h):  #
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)

def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')   #
    interp.load_state_dict(states['state_dictAT'])    #
    flow.load_state_dict(states['state_dictFC'])

def interpolate_batch(frames, factor):  #
    frame0 = torch.stack(frames[:-1])   #
    frame1 = torch.stack(frames[1:])    #

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)    #

    flow_out = flow(ix)   #
    f01 = flow_out[:, :2, :, :]  #
    f10 = flow_out[:, 2:, :, :]  #

    frame_buffer = []
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
        vt0 = F.sigmoid(io[:, 4:5, :, :])  #
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)   #
        gi1ft1f = back_warp(i1, ft1f)   #

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)  #

        frame_buffer.append(ft_p)

    return frame_buffer

##
def load_batch(file_in, batch_size, batch, w, h):  #
    if len(batch) > 0:
        batch = [batch[-1]]    #

    imgnamelist = os.listdir(file_in)
    for i in range(batch_size):
        frame = cv2.imread(file_in + imgnamelist[i])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)    #
        frame = frame.resize((w, h), Image.ANTIALIAS)  #
        frame = frame.convert('RGB')
        frame = trans_forward(frame)   #
        batch.append(frame)   #

    return batch

def denorm_frame(frame, w0, h0):
    frame = frame.cpu()
    frame = trans_backward(frame)  #
    frame = frame.resize((w0, h0), Image.BILINEAR)
    frame = frame.convert('RGB')
    return np.array(frame)[:, :, ::-1].copy()

def generate_img(file_in, file_out, factor, batch_size=10):
    imgnamelist = os.listdir(file_in)
    imtmp = cv2.imread(file_in + imgnamelist[0])
    w0, h0, c0 = imtmp.shape

    w, h = (w0 // 32) * 32, (h0 // 32) * 32  #
    setup_back_warp(w, h)

    batch = []
    batch = load_batch(file_in, batch_size, batch, w, h)
    if len(batch) == 1:  #
        print("图像数目不够!")
        return 0

    intermediate_frames = interpolate_batch(batch, factor)
    intermediate_frames = list(zip(*intermediate_frames))  #

    for fid, iframe in enumerate(intermediate_frames):
        for frm in iframe:
            cv2.imwrite(file_out + 'insertion.jpg', denorm_frame(frm, w0, h0))

def main(file_in, checkpoint, file_out, batch, factor):
    load_models(checkpoint)
    generate_img(file_in, file_out, batch, factor)

if __name__ == '__main__':
    main('./data/test/', './save/checkpoints/SuperSloMo29.ckpt', './data/test/', 2, 2)


