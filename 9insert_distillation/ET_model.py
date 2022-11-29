import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        #
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x

class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn=None):
        #
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        if skpCn != None:
            # Convolution + Leaky ReLU on (`x`, `skpCn`)
            x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x

class ET_model(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(ET_model, self).__init__()
        #
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 128, 5)
        self.down2 = down(128, 256, 3)
        self.up1 = up(256, 128)
        self.up2 = up(128, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)     # inChannels  32    batchsize*inChannels*352*352   batchsize*32*352*352
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)    # 32  32    batchsize*32*352*352   batchsize*32*352*352
        s2 = self.down1(s1)    # batchsize*32*352*352   batchsize*128*176*176
        x = self.down2(s2)    # batchsize*128*176*176  batchsize*256*88*88
        # x = self.conv2(x)     # batchsize*256*88*88  batchsize*256*88*88
        x = self.up1(x, s2)   # batchsize*256*88*88和batchsize*128*176*176   batchsize*128*176*176
        x = self.up2(x, s1)   # batchsize*128*176*176和batchsize*32*352*352  batchsize*32*352*352
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x

class backWarp(nn.Module):
    def __init__(self, W, H, device):
        super(backWarp, self).__init__()
        # 创建一个网格
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1 （
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        #
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)
        return imgOut

class Get_Coeff:
    def __init__(self, t_st=0, t_ed=1, t_num=7, device=torch.device("cuda:0")):   # t_st=0.125, t_ed=0.875, t_num=7
        t_tmp = np.linspace(t_st, t_ed, t_num + 2)
        self.t = t_tmp[1:t_num+1]
        self.device = device

    def getFlowCoeff (self, indices):
        #
        ind = indices.detach().numpy()
        C11 = C00 = - (1 - (self.t[ind])) * (self.t[ind])
        C01 = (self.t[ind]) * (self.t[ind])
        C10 = (1 - (self.t[ind])) * (1 - (self.t[ind]))
        return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C01)[None, None, None,
                                                                                      :].permute(3, 0, 1, 2).to(
            self.device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C11)[None, None,
                                                                                            None, :].permute(3, 0, 1, 2).to(
            self.device)

    def getWarpCoeff (self, indices):
        #
        ind = indices.detach().numpy()
        C0 = 1 - self.t[ind]
        C1 = self.t[ind]
        return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(self.device)

class Get_Coeff_direct:
    def __init__(self, device=torch.device("cuda:0")):   # t_st=0.125, t_ed=0.875, t_num=7
        self.device = device

    def getFlowCoeff (self, radio):
        #
        radio = radio.detach().numpy()
        C11 = C00 = - (1 - radio) * radio
        C01 = radio * radio
        C10 = (1 - radio) * (1 - radio)
        return torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C01)[None, None, None,
                                                                                      :].permute(3, 0, 1, 2).to(
            self.device), torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C11)[None, None,
                                                                                            None, :].permute(3, 0, 1, 2).to(
            self.device)

    def getWarpCoeff (self, radio):
        radio = radio.detach().numpy()
        C0 = 1 - radio
        C1 = radio
        return torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(self.device), torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(self.device)
