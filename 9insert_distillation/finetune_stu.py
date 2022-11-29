import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import model
import dataloader
from math import log10
import time
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from tifffile import imread

import os
import ET_model
import ET_fineturn_dataloader

# 用于获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(args, flowComp, ArbTimeFlowIntrp, validationFlowBackWarp, get_coeff, vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn):
    psnr = 0
    tloss = 0
    flag = 1
    with torch.no_grad():  # 不进行反向传播
        for validationIndex, (validationData, validationFrameradio) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(args.device)
            I1 = frame1.to(args.device)
            IFrame = frameT.to(args.device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))  #
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = get_coeff.getFlowCoeff(validationFrameradio)  #

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0  #
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)  #
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

            intrpOut = ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))  #

            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0  #
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])  #
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)  #
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)

            wCoeff = get_coeff.getWarpCoeff(validationFrameradio)  #

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            # For tensorboard
            if (flag):
                retImg = torchvision.utils.make_grid(
                    [frame0[0], frameT[0], Ft_p.cpu()[0], frame1[0]], padding=10)
                flag = 0

            # loss
            recnLoss = L1_lossFn(Ft_p, IFrame)  #

            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p),
                                  vgg16_conv_4_3(IFrame))  #

            #
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
                validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)

            #
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()

            # psnr
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1 / MSE_val.item()))

    return (psnr / len(validationloader)), (tloss / len(validationloader)), retImg

def student_fineturne(args, flowComp, ArbTimeFlowIntrp, vgg16_conv_4_3,
                     trainFlowBackWarp, validationFlowBackWarp,
                     trainloader, validationloader,
                     L1_lossFn, MSE_LossFn, optimizer, scheduler, writer):
    ##
    dict_student = torch.load(args.checkpoint_dir + "/student_checkpoints/SuperSloMo_stu_039.ckpt")
    flowComp.load_state_dict(dict_student['state_dictFC'])
    ArbTimeFlowIntrp.load_state_dict(dict_student['state_dictAT'])

    dict1 = {'checkpoint_counter': 0, 'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}
    cLoss = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']
    checkpoint_counter = dict1['checkpoint_counter']

    ##
    get_coeff = ET_model.Get_Coeff_direct(device=args.device)

    #
    start = time.time()
    for epoch in range(dict1['epoch'] + 1, args.epochs):
        print("Epoch: ", epoch)

        #
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        optimizer.step()
        #
        scheduler.step()

        for trainIndex, (trainData, trainFrameRadio) in enumerate(trainloader, 0):
            #
            frame0, frameT, frame1 = trainData
            I0 = frame0.to(args.device)  #
            I1 = frame1.to(args.device)  #
            IFrame = frameT.to(args.device)  #

            optimizer.zero_grad()  #

            #
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            #
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            #
            fCoeff = get_coeff.getFlowCoeff(trainFrameRadio)
            #
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
            #
            g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)

            #
            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
            #
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0
            #
            g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
            #
            wCoeff = get_coeff.getWarpCoeff(trainFrameRadio)
            #
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                    wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            #
            recnLoss = L1_lossFn(Ft_p, IFrame)  #
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))  #
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
                trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)  #
            #
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1  #
            #
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth  # 204 102

            #
            loss.backward()
            optimizer.step()
            iLoss += loss.item()  #

            #
            if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
                end = time.time()

                psnr, vLoss, valImg = validate(args, flowComp, ArbTimeFlowIntrp, validationFlowBackWarp, get_coeff,
                                               vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn)

                valPSNR[epoch].append(psnr)
                valLoss[epoch].append(vLoss)

                # Tensorboard
                itr = trainIndex + epoch * (len(trainloader))
                writer.add_scalars('Loss', {'trainLoss': iLoss / args.progress_iter,  # 计算每个args.progress_iter内的平均损失
                                            'validationLoss': vLoss}, itr)
                writer.add_scalar('PSNR', psnr, itr)
                writer.add_image('Validation', valImg, itr)
                #####

                endVal = time.time()
                print(
                    " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (
                        iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr,
                        endVal - end,
                        get_lr(optimizer)))
                cLoss[epoch].append(iLoss / args.progress_iter)
                iLoss = 0
                start = time.time()

        #
        if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):  # args.checkpoint_epoch - 1
            dict1 = {
                'Detail': "End to end Super_SloMo_stu.",
                'checkpoint_counter': checkpoint_counter,
                'epoch': epoch,
                'timestamp': datetime.datetime.now(),
                'trainBatchSz': args.train_batch_size,
                'validationBatchSz': args.validation_batch_size,
                'learningRate': get_lr(optimizer),
                'loss': cLoss,
                'valLoss': valLoss,
                'valPSNR': valPSNR,
                'state_dictFC': flowComp.state_dict(),
                'state_dictAT': ArbTimeFlowIntrp.state_dict(),
            }
            torch.save(dict1, args.checkpoint_dir + "/student_fineturne/SuperSloMo_fine_" + str(checkpoint_counter).zfill(
                3) + ".ckpt")
            checkpoint_counter += 1

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--dataset_root", type=str, default='./data/ribbon/',
                        help='path to dataset folder containing train-test-validation folders')      # ./data/ETdata/
    parser.add_argument("--checkpoint_dir", type=str, default='./save/',
                        help='path to folder for saving checkpoints')
    parser.add_argument("--train_continue", type=bool, default=False,
                        help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
    parser.add_argument("--epochs", type=int, default=50, help='number of epochs to train. Default: 200.')  # synapix 50
    parser.add_argument("--train_batch_size", type=int, default=3, help='batch size for training. Default: 6.') #synapix 4
    parser.add_argument("--validation_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
    parser.add_argument("--init_learning_rate", type=float, default=0.000001,
                        help='set initial learning rate. Default: 0.0001.')
    parser.add_argument("--milestones", type=list, default=[100, 150],
                        help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
    parser.add_argument("--progress_iter", type=int, default=10,
                        help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
    parser.add_argument("--checkpoint_epoch", type=int, default=1,
                        help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
    args = parser.parse_args()

    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ##
    imgnamelist_w_h = os.listdir(args.dataset_root + "/train/000/")
    imtmp = imread(args.dataset_root + "/train/000/" + imgnamelist_w_h[0])
    if len(imtmp.shape) == 2:
        imtmp = np.expand_dims(imtmp, axis=2)
        imtmp = np.concatenate((imtmp, imtmp, imtmp), axis=-1)
    w0, h0 = imtmp[:,:,0].shape
    w, h = (w0 // 32 - 40) * 32, (h0 // 32 - 40) * 32  #

    transform = transforms.Compose([transforms.ToTensor()])
    TP = transforms.Compose([transforms.ToPILImage()])

    trainset = ET_fineturn_dataloader.ET(root=args.dataset_root + '/train', transform=transform,
                                         fixsize=(w, h), train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    validationset = ET_fineturn_dataloader.ET(root=args.dataset_root + '/val', transform=transform,
                                          fixsize=(w, h), train=False)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)
    # print(trainset, validationset)

    ##
    vgg16 = torchvision.models.vgg16(pretrained=True)  #
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])  #
    vgg16_conv_4_3.to(device)
    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False  #

    ##
    student_flowComp = ET_model.ET_model(6,
                                         4)  #
    student_flowComp.to(device)
    student_ArbTimeFlowIntrp = ET_model.ET_model(20, 5)  #
    student_ArbTimeFlowIntrp.to(device)
    #
    trainFlowBackWarp = ET_model.backWarp(w, h, device)  #
    trainFlowBackWarp = trainFlowBackWarp.to(device)
    validationFlowBackWarp = ET_model.backWarp(w, h, device)
    validationFlowBackWarp = validationFlowBackWarp.to(device)

    ##
    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()
    params = list(student_flowComp.parameters()) + list(student_ArbTimeFlowIntrp.parameters())
    optimizer = optim.Adam(params, lr=args.init_learning_rate)
    #
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    writer = SummaryWriter('log_fineturn')

    student_fineturne(args, student_flowComp, student_ArbTimeFlowIntrp, vgg16_conv_4_3,
                      trainFlowBackWarp, validationFlowBackWarp,
                      trainloader, validationloader,
                      L1_lossFn, MSE_LossFn, optimizer, scheduler, writer)