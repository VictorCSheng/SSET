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

import os
import ET_model

#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

##
def validate(args, flowComp, ArbTimeFlowIntrp, validationFlowBackWarp, vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn):
    psnr = 0
    tloss = 0
    flag = 1
    with torch.no_grad():  #
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(args.device)
            I1 = frame1.to(args.device)
            IFrame = frameT.to(args.device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))  #
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            fCoeff = model.getFlowCoeff(validationFrameIndex, args.device)  #

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

            wCoeff = model.getWarpCoeff(validationFrameIndex, args.device)  #

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            #
            if (flag):
                retImg = torchvision.utils.make_grid(
                    [args.revNormalize(frame0[0]), args.revNormalize(frameT[0]), args.revNormalize(Ft_p.cpu()[0]),
                     args.revNormalize(frame1[0])], padding=10)
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

def train(args, flowComp, ArbTimeFlowIntrp, trainFlowBackWarp, trainloader, validationFlowBackWarp,
              validationloader, vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer):
    ##
    if args.train_continue:
        checkpoint_files = os.listdir(args.checkpoint_dir + "/teacher_checkpoints/")
        checkpoint_file_path = os.path.join(args.checkpoint_dir + "/teacher_checkpoints/", checkpoint_files[-1])
        dict_stu = torch.load(checkpoint_file_path)
        student_flowComp.load_state_dict(dict_stu['state_dictFC'])
        student_ArbTimeFlowIntrp.load_state_dict(dict_stu['state_dictAT'])
    else:
        dict1 = {'checkpoint_counter': 0, 'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    cLoss = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']
    checkpoint_counter = dict1['checkpoint_counter']

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

        for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
            #
            frame0, frameT, frame1 = trainData
            I0 = frame0.to(args.device)      #
            I1 = frame1.to(args.device)      #
            IFrame = frameT.to(args.device)  #

            optimizer.zero_grad()  #

            #
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            #
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            #
            fCoeff = model.getFlowCoeff(trainFrameIndex, args.device)
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
            wCoeff = model.getWarpCoeff(trainFrameIndex, args.device)
            #
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

            #
            recnLoss = L1_lossFn(Ft_p, IFrame)  #
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))  #
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
                trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)    #
            #
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1   #
            #
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            #
            loss.backward()
            optimizer.step()
            iLoss += loss.item()    #

            #
            if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
                end = time.time()

                psnr, vLoss, valImg = validate(args, flowComp, ArbTimeFlowIntrp, validationFlowBackWarp,
                                               vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn)

                valPSNR[epoch].append(psnr)
                valLoss[epoch].append(vLoss)

                # Tensorboard
                itr = trainIndex + epoch * (len(trainloader))
                writer.add_scalars('Loss', {'trainLoss': iLoss / args.progress_iter,    #计算每个args.progress_iter内的平均损失
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
            torch.save(dict1, args.checkpoint_dir + "/student_checkpoints/SuperSloMo_stu_" + str(checkpoint_counter).zfill(3) + ".ckpt")
            checkpoint_counter += 1

"""
"""
def train_distillation(args, trainloader, validationloader,
                       teacher_flowComp, teacher_ArbTimeFlowIntrp,
                       student_flowComp, student_ArbTimeFlowIntrp,
                       trainFlowBackWarp, validationFlowBackWarp,
                       vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer):
    ##
    dict_teacher = torch.load(args.checkpoint_dir + "/teacher_checkpoints/SuperSloMo.ckpt")
    teacher_flowComp.load_state_dict(dict_teacher['state_dictFC'])
    teacher_ArbTimeFlowIntrp.load_state_dict(dict_teacher['state_dictAT'])
    for param in teacher_flowComp.parameters():
        param.requires_grad = False  #
    for param in teacher_ArbTimeFlowIntrp.parameters():
        param.requires_grad = False  #
    ##
    if args.train_continue:
        checkpoint_files = os.listdir(args.checkpoint_dir + "/student_checkpoints/")
        checkpoint_file_path = os.path.join(args.checkpoint_dir + "/student_checkpoints/", checkpoint_files[-1])
        dict_stu = torch.load(checkpoint_file_path)
        student_flowComp.load_state_dict(dict_stu['state_dictFC'])
        student_ArbTimeFlowIntrp.load_state_dict(dict_stu['state_dictAT'])
    else:
        dict_stu = {'checkpoint_counter': 0, 'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    cLoss = dict_stu['loss']
    valLoss = dict_stu['valLoss']
    valPSNR = dict_stu['valPSNR']
    checkpoint_counter = dict_stu['checkpoint_counter']

    ##
    get_coeff = ET_model.Get_Coeff(device = args.device)
    ##
    loss_dis = nn.KLDivLoss()

    #
    start = time.time()
    for epoch in range(dict_stu['epoch'] + 1, args.epochs):
        print("Epoch: ", epoch)
        #
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        optimizer.step()
        #
        scheduler.step()

        for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
            #
            frame0, frameT, frame1 = trainData
            I0 = frame0.to(args.device)  #
            I1 = frame1.to(args.device)  #
            IFrame = frameT.to(args.device)  #

            optimizer.zero_grad()  #

            ## student_model
            flowOut_stu = student_flowComp(torch.cat((I0, I1), dim=1))
            #
            F_0_1_stu = flowOut_stu[:, :2, :, :]
            F_1_0_stu = flowOut_stu[:, 2:, :, :]
            #
            fCoeff = get_coeff.getFlowCoeff(trainFrameIndex)
            #
            F_t_0_stu = fCoeff[0] * F_0_1_stu + fCoeff[1] * F_1_0_stu
            F_t_1_stu = fCoeff[2] * F_0_1_stu + fCoeff[3] * F_1_0_stu
            #
            g_I0_F_t_0_stu = trainFlowBackWarp(I0, F_t_0_stu)
            g_I1_F_t_1_stu = trainFlowBackWarp(I1, F_t_1_stu)

            #
            intrpOut_stu = student_ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1_stu, F_1_0_stu, F_t_1_stu, F_t_0_stu, g_I1_F_t_1_stu, g_I0_F_t_0_stu), dim=1))
            #
            F_t_0_f_stu = intrpOut_stu[:, :2, :, :] + F_t_0_stu
            F_t_1_f_stu = intrpOut_stu[:, 2:4, :, :] + F_t_1_stu
            V_t_0_stu = torch.sigmoid(intrpOut_stu[:, 4:5, :, :])
            V_t_1_stu = 1 - V_t_0_stu
            #
            g_I0_F_t_0_f_stu = trainFlowBackWarp(I0, F_t_0_f_stu)
            g_I1_F_t_1_f_stu = trainFlowBackWarp(I1, F_t_1_f_stu)
            #
            wCoeff = get_coeff.getWarpCoeff(trainFrameIndex)
            #
            Ft_p_stu = (wCoeff[0] * V_t_0_stu * g_I0_F_t_0_f_stu + wCoeff[1] * V_t_1_stu * g_I1_F_t_1_f_stu) / (
                    wCoeff[0] * V_t_0_stu + wCoeff[1] * V_t_1_stu)

            #
            recnLoss = L1_lossFn(Ft_p_stu, IFrame)  #
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p_stu), vgg16_conv_4_3(IFrame))  #
            warpLoss = L1_lossFn(g_I0_F_t_0_f_stu, IFrame) + L1_lossFn(g_I1_F_t_1_f_stu, IFrame) + L1_lossFn(
                trainFlowBackWarp(I0, F_1_0_stu), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1_stu), I0)  #
            #
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0_stu[:, :, :, :-1] - F_1_0_stu[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0_stu[:, :, :-1, :] - F_1_0_stu[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1_stu[:, :, :, :-1] - F_0_1_stu[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1_stu[:, :, :-1, :] - F_0_1_stu[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1  #
            #
            loss_stu = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            ## teacher_model
            #
            flowOut_tea = teacher_flowComp(torch.cat((I0, I1), dim=1))  #   1

            #
            F_0_1_tea = flowOut_tea[:, :2, :, :]
            F_1_0_tea = flowOut_tea[:, 2:, :, :]
            #
            F_t_0_tea = fCoeff[0] * F_0_1_tea + fCoeff[1] * F_1_0_tea
            F_t_1_tea = fCoeff[2] * F_0_1_tea + fCoeff[3] * F_1_0_tea
            #
            g_I0_F_t_0_tea = trainFlowBackWarp(I0, F_t_0_tea)
            g_I1_F_t_1_tea = trainFlowBackWarp(I1, F_t_1_tea)
            intrpOut_tea = teacher_ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1_tea, F_1_0_tea, F_t_1_tea, F_t_0_tea, g_I1_F_t_1_tea, g_I0_F_t_0_tea), dim=1))   # 2

            ##
            loss_flow = L1_lossFn(flowOut_stu / args.T_distillation, flowOut_tea / args.T_distillation)
            loss_intrpOut = L1_lossFn(intrpOut_stu / args.T_distillation, intrpOut_tea / args.T_distillation)

            ##
            loss_sum = loss_stu + args.T_distillation * args.T_distillation * (0.5 * loss_flow + 0.5 * loss_intrpOut)

            #
            loss_sum.backward()
            optimizer.step()
            iLoss += loss_sum.item()  #

            #
            if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
                end = time.time()

                psnr, vLoss, valImg = validate(args, student_flowComp, student_ArbTimeFlowIntrp, validationFlowBackWarp,
                                               vgg16_conv_4_3, validationloader,L1_lossFn, MSE_LossFn)

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
                'state_dictFC': student_flowComp.state_dict(),
                'state_dictAT': student_ArbTimeFlowIntrp.state_dict(),
            }
            torch.save(dict1, args.checkpoint_dir + "/student_checkpoints/SuperSloMo_stu_" + str(
                checkpoint_counter).zfill(3) + ".ckpt")
            checkpoint_counter += 1

def train_distillation_flowComp(args, trainloader, validationloader,
                       teacher_flowComp, teacher_ArbTimeFlowIntrp,
                       student_flowComp,
                       trainFlowBackWarp, validationFlowBackWarp,
                       vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer):
    ##
    dict_teacher = torch.load(args.checkpoint_dir + "/teacher_checkpoints/SuperSloMo.ckpt")
    teacher_flowComp.load_state_dict(dict_teacher['state_dictFC'])
    teacher_ArbTimeFlowIntrp.load_state_dict(dict_teacher['state_dictAT'])
    for param in teacher_flowComp.parameters():
        param.requires_grad = False  #
    for param in teacher_ArbTimeFlowIntrp.parameters():
        param.requires_grad = False  #
    ##
    if args.train_continue:
        checkpoint_files = os.listdir(args.checkpoint_dir + "/student_checkpoints/")
        checkpoint_file_path = os.path.join(args.checkpoint_dir + "/student_checkpoints/", checkpoint_files[-1])
        dict_stu = torch.load(checkpoint_file_path)
        student_flowComp.load_state_dict(dict_stu['state_dictFC'])
        student_ArbTimeFlowIntrp.load_state_dict(dict_stu['state_dictAT'])
    else:
        dict_stu = {'checkpoint_counter': 0, 'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    cLoss = dict_stu['loss']
    valLoss = dict_stu['valLoss']
    valPSNR = dict_stu['valPSNR']
    checkpoint_counter = dict_stu['checkpoint_counter']

    ##
    get_coeff = ET_model.Get_Coeff(device=args.device)
    ##
    loss_dis = nn.KLDivLoss()

    #
    start = time.time()
    for epoch in range(dict_stu['epoch'] + 1, args.epochs):
        print("Epoch: ", epoch)
        #
        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iLoss = 0

        optimizer.step()
        #
        scheduler.step()

        for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
            #
            frame0, frameT, frame1 = trainData
            I0 = frame0.to(args.device)  #
            I1 = frame1.to(args.device)  #
            IFrame = frameT.to(args.device)  #
            optimizer.zero_grad()  #

            ## student_model
            #
            flowOut_stu = student_flowComp(torch.cat((I0, I1), dim=1))
            #
            F_0_1_stu = flowOut_stu[:, :2, :, :]
            F_1_0_stu = flowOut_stu[:, 2:, :, :]
            #
            fCoeff = get_coeff.getFlowCoeff(trainFrameIndex)
            #
            F_t_0_stu = fCoeff[0] * F_0_1_stu + fCoeff[1] * F_1_0_stu
            F_t_1_stu = fCoeff[2] * F_0_1_stu + fCoeff[3] * F_1_0_stu
            #
            g_I0_F_t_0_stu = trainFlowBackWarp(I0, F_t_0_stu)
            g_I1_F_t_1_stu = trainFlowBackWarp(I1, F_t_1_stu)

            #
            intrpOut_stu = teacher_ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1_stu, F_1_0_stu, F_t_1_stu, F_t_0_stu, g_I1_F_t_1_stu, g_I0_F_t_0_stu), dim=1))
            #
            F_t_0_f_stu = intrpOut_stu[:, :2, :, :] + F_t_0_stu
            F_t_1_f_stu = intrpOut_stu[:, 2:4, :, :] + F_t_1_stu
            V_t_0_stu = torch.sigmoid(intrpOut_stu[:, 4:5, :, :])
            V_t_1_stu = 1 - V_t_0_stu
            #
            g_I0_F_t_0_f_stu = trainFlowBackWarp(I0, F_t_0_f_stu)
            g_I1_F_t_1_f_stu = trainFlowBackWarp(I1, F_t_1_f_stu)
            #
            wCoeff = get_coeff.getWarpCoeff(trainFrameIndex)
            #
            Ft_p_stu = (wCoeff[0] * V_t_0_stu * g_I0_F_t_0_f_stu + wCoeff[1] * V_t_1_stu * g_I1_F_t_1_f_stu) / (
                    wCoeff[0] * V_t_0_stu + wCoeff[1] * V_t_1_stu)

            #
            recnLoss = L1_lossFn(Ft_p_stu, IFrame)  #
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p_stu), vgg16_conv_4_3(IFrame))  #
            warpLoss = L1_lossFn(g_I0_F_t_0_f_stu, IFrame) + L1_lossFn(g_I1_F_t_1_f_stu, IFrame) + L1_lossFn(
                trainFlowBackWarp(I0, F_1_0_stu), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1_stu),
                                                                  I0)  #
            #
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0_stu[:, :, :, :-1] - F_1_0_stu[:, :, :, 1:])) + torch.mean(
                torch.abs(F_1_0_stu[:, :, :-1, :] - F_1_0_stu[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1_stu[:, :, :, :-1] - F_0_1_stu[:, :, :, 1:])) + torch.mean(
                torch.abs(F_0_1_stu[:, :, :-1, :] - F_0_1_stu[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1  #
            #
            loss_stu = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

            ## teacher_model
            #
            flowOut_tea = teacher_flowComp(torch.cat((I0, I1), dim=1))  # 1

            #
            F_0_1_tea = flowOut_tea[:, :2, :, :]
            F_1_0_tea = flowOut_tea[:, 2:, :, :]
            #
            F_t_0_tea = fCoeff[0] * F_0_1_tea + fCoeff[1] * F_1_0_tea
            F_t_1_tea = fCoeff[2] * F_0_1_tea + fCoeff[3] * F_1_0_tea
            #
            g_I0_F_t_0_tea = trainFlowBackWarp(I0, F_t_0_tea)
            g_I1_F_t_1_tea = trainFlowBackWarp(I1, F_t_1_tea)
            intrpOut_tea = teacher_ArbTimeFlowIntrp(
                torch.cat((I0, I1, F_0_1_tea, F_1_0_tea, F_t_1_tea, F_t_0_tea, g_I1_F_t_1_tea, g_I0_F_t_0_tea),
                          dim=1))  # 2

            ##
            loss_flow = L1_lossFn(flowOut_stu / args.T_distillation, flowOut_tea / args.T_distillation)
            loss_intrpOut = L1_lossFn(intrpOut_stu / args.T_distillation, intrpOut_tea / args.T_distillation)

            ##
            loss_sum = loss_stu + args.T_distillation * args.T_distillation * (0.5 * loss_flow + 0.5 * loss_intrpOut)

            #
            loss_sum.backward()
            optimizer.step()
            iLoss += loss_sum.item()  #

            #
            if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
                end = time.time()

                psnr, vLoss, valImg = validate(args, student_flowComp, teacher_ArbTimeFlowIntrp, validationFlowBackWarp,
                                               vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn)

                valPSNR[epoch].append(psnr)
                valLoss[epoch].append(vLoss)

                # Tensorboard
                itr = trainIndex + epoch * (len(trainloader))
                writer.add_scalars('Loss', {'trainLoss': iLoss / args.progress_iter,  #
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
                'state_dictFC': student_flowComp.state_dict(),
                'state_dictAT': student_ArbTimeFlowIntrp.state_dict(),
            }
            torch.save(dict1, args.checkpoint_dir + "/student_checkpoints/SuperSloMo_stu_" + str(
                checkpoint_counter).zfill(3) + ".ckpt")
            checkpoint_counter += 1

def teachercompare(args, flowComp, ArbTimeFlowIntrp, vgg16_conv_4_3,
                   trainFlowBackWarp, validationFlowBackWarp,
                   trainloader, validationloader,
                   L1_lossFn, MSE_LossFn):
    ##
    dict_teacher = torch.load(args.checkpoint_dir + "/teacher_checkpoints/SuperSloMo.ckpt")
    flowComp.load_state_dict(dict_teacher['state_dictFC'])
    ArbTimeFlowIntrp.load_state_dict(dict_teacher['state_dictAT'])

    iLoss = 0
    start = time.time()
    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):
        #
        frame0, frameT, frame1 = trainData
        I0 = frame0.to(args.device)  #
        I1 = frame1.to(args.device)  #
        IFrame = frameT.to(args.device)  #

        #
        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        #
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        #
        fCoeff = model.getFlowCoeff(trainFrameIndex, args.device)
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
        wCoeff = model.getWarpCoeff(trainFrameIndex, args.device)
        #
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        #
        recnLoss = L1_lossFn(Ft_p, IFrame)  #
        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))  #
        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
            trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)  #
        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
            torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
            torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1  #
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
        iLoss += loss.item()  #

        if ((trainIndex % args.progress_iter) == args.progress_iter - 1):
            end = time.time()

            psnr, vLoss, valImg = validate(args, flowComp, ArbTimeFlowIntrp, validationFlowBackWarp,
                                           vgg16_conv_4_3, validationloader, L1_lossFn, MSE_LossFn)

            endVal = time.time()
            print(
                " Loss: %0.6f  Iterations: %4d/%4d  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f LearningRate: %f" % (
                    iLoss / args.progress_iter, trainIndex, len(trainloader), end - start, vLoss, psnr,
                    endVal - end,
                    get_lr(optimizer)))
            iLoss = 0
            start = time.time()

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--dataset_root", type=str, default='./data/adobedata/',
                        help='path to dataset folder containing train-test-validation folders')
    parser.add_argument("--checkpoint_dir", type=str, default='./save/',
                        help='path to folder for saving checkpoints')
    parser.add_argument("--train_continue", type=bool, default=False,
                        help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
    parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
    parser.add_argument("--train_batch_size", type=int, default=12, help='batch size for training. Default: 6.')
    parser.add_argument("--validation_batch_size", type=int, default=10, help='batch size for validation. Default: 10.')
    parser.add_argument("--init_learning_rate", type=float, default=0.0001,
                        help='set initial learning rate. Default: 0.0001.')
    parser.add_argument("--milestones", type=list, default=[100, 150],
                        help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
    parser.add_argument("--progress_iter", type=int, default=100,
                        help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
    parser.add_argument("--T_distillation", type=int, default=2,
                        help='Temperature of distillation network.')
    parser.add_argument("--checkpoint_epoch", type=int, default=5,
                        help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
    args = parser.parse_args()

    #
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device
    #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ##
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    #
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    args.revNormalize = revNormalize

    trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transform=transform, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)   #

    validationset = dataloader.SuperSloMo(root=args.dataset_root + '/validation', transform=transform, randomCropSize=(640, 352), train=False)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)
    # print(trainset, validationset)

    ##
    vgg16 = torchvision.models.vgg16(pretrained=True)  #
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])  #
    vgg16_conv_4_3.to(device)
    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False  #

    ## teacher网络设置
    #
    teacher_flowComp = model.UNet(6, 4)  #
    teacher_flowComp.to(device)
    teacher_ArbTimeFlowIntrp = model.UNet(20, 5)  #
    teacher_ArbTimeFlowIntrp.to(device)

    ##
    student_flowComp = ET_model.ET_model(6, 4)  #
    student_flowComp.to(device)
    student_ArbTimeFlowIntrp = ET_model.ET_model(20, 5)  #
    student_ArbTimeFlowIntrp.to(device)
    #
    trainFlowBackWarp = ET_model.backWarp(352, 352, device)    #
    trainFlowBackWarp = trainFlowBackWarp.to(device)
    validationFlowBackWarp = ET_model.backWarp(640, 352, device)
    validationFlowBackWarp = validationFlowBackWarp.to(device)

    ##
    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()
    params = list(student_flowComp.parameters()) + list(student_ArbTimeFlowIntrp.parameters())
    # params = list(student_flowComp.parameters())
    optimizer = optim.Adam(params, lr=args.init_learning_rate)
    #
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    writer = SummaryWriter('log')

    # train(args, teacher_flowComp, teacher_ArbTimeFlowIntrp, trainFlowBackWarp, trainloader, validationFlowBackWarp,
    #       validationloader, vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer)

    # train(args, student_flowComp, student_ArbTimeFlowIntrp, trainFlowBackWarp, trainloader, validationFlowBackWarp,
    #       validationloader, vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer)

    train_distillation(args, trainloader, validationloader,
                       teacher_flowComp, teacher_ArbTimeFlowIntrp,
                       student_flowComp, student_ArbTimeFlowIntrp,
                       trainFlowBackWarp, validationFlowBackWarp,
                       vgg16_conv_4_3, L1_lossFn, MSE_LossFn, optimizer, scheduler, writer)

    # teachercompare(args, teacher_flowComp, teacher_ArbTimeFlowIntrp, vgg16_conv_4_3,
    #                trainFlowBackWarp, validationFlowBackWarp,
    #                trainloader, validationloader,
    #                L1_lossFn, MSE_LossFn)
