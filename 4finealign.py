import os
import cv2
import copy
import argparse
import numpy as np
import scipy.io as scio

import dgl
import torch

from csutil import util
from csutil.csgraph import SmallareaTemplatejMatch, Elastic_Graph
from csutil.csBP import run_BP
from csutil.cselastic import updata_all_example

if __name__ == '__main__':
    #
    scale_factorreg = 4
    constant_scalfac = 4
    #
    parser_graph = argparse.ArgumentParser(description='SmallareaTemplatejMatch parameter')
    parser_graph.add_argument('--bigrow', default=6, type=int, help='bigrow')   # 网格是3*3的
    parser_graph.add_argument('--bigcol', default=6, type=int, help='bigcol')
    parser_graph.add_argument('--Elasticiter', default=5, type=int, help='Number of Elastic Model Iterations')  # HUA之前的都为5
    parser_graph.add_argument('--dst_radius', default=25, type=int, help='dst_radius')  ##5:35 45   4:40 50 ribon:60  hua1:4（网格） 20  hua2:5 17
    parser_graph.add_argument('--scale_factor', default=8, type=int, help='scale_factor')

    parser_BP = argparse.ArgumentParser(description='BP parameter')
    parser_BP.add_argument('--TMscore_thresholdL', default=0.15, type=int, help='TMscore_thresholdL')
    parser_BP.add_argument('--numIter', default=8, type=int, help='numIter')  ##BPparameter.numIter = 10
    parser_BP.add_argument('--bplambda', default=0.1, type=int, help='bplambda')  ### 0.05 datacost所占的比重
    parser_BP.add_argument('--data_k', default=200, type=int, help='data_k')  ### 10  ############datacost截断阈值
    parser_BP.add_argument('--disc_k', default=50, type=int, help='disc_k')  ### 200  ############信息传递中的截断阈值

    TEMs_coarse_b_dir = './img/TEMs/finetemp/'
    TEMs_fine_dir = './img/TEMs/fine/'

    TOM_coarse_b_dir = './img/TOM/finetemp/'
    TOM_fine_dir = './img/TOM/fine/'

    TEMs_reg_config_dir = './reg_config/TEMs/'

    util.mkdir(TEMs_fine_dir)
    util.mkdir(TOM_fine_dir)
    util.mkdir(TEMs_reg_config_dir)

    imgnamelist = os.listdir(TEMs_coarse_b_dir)
    imgnamelist = sorted(imgnamelist)

    imgtemp = cv2.imread(TEMs_coarse_b_dir + imgnamelist[0])
    if imgtemp.ndim == 3:
        imgtemp = cv2.cvtColor(imgtemp, cv2.COLOR_RGB2GRAY)
    lefttopx_new = 0
    lefttopy_new = 0
    rightbottomx_new = imgtemp.shape[0]
    rightbottomy_new = imgtemp.shape[1]

    Graph_parameter = parser_graph.parse_args()
    Graph_parameter.lefttopx = int(lefttopx_new / Graph_parameter.scale_factor)
    Graph_parameter.lefttopy = int(lefttopy_new / Graph_parameter.scale_factor)
    Graph_parameter.rightbottomx = int(rightbottomx_new / Graph_parameter.scale_factor)
    Graph_parameter.rightbottomy = int(rightbottomy_new / Graph_parameter.scale_factor)
    Graph_parameter.grid_radius_x = (Graph_parameter.rightbottomx - Graph_parameter.lefttopx) // (
            2 * Graph_parameter.bigrow)
    Graph_parameter.grid_radius_y = (Graph_parameter.rightbottomy - Graph_parameter.lefttopy) // (
            2 * Graph_parameter.bigcol)
    Graph_parameter.search_radius = min(Graph_parameter.grid_radius_x, Graph_parameter.grid_radius_y) - 5   # 搜索半径

    for i in range(len(imgnamelist) - 1):
        if i == 0:
            img1ori = cv2.imread(TEMs_coarse_b_dir + imgnamelist[i])
            if img1ori.ndim == 3:
                img1ori = cv2.cvtColor(img1ori, cv2.COLOR_RGB2GRAY)
        else:
            img1ori = cv2.imread(TEMs_fine_dir + imgnamelist[i])
            if img1ori.ndim == 3:
                img1ori = cv2.cvtColor(img1ori, cv2.COLOR_RGB2GRAY)

        img2ori = cv2.imread(TEMs_coarse_b_dir + imgnamelist[i + 1])
        if img2ori.ndim == 3:
            img2ori = cv2.cvtColor(img2ori, cv2.COLOR_RGB2GRAY)

        #
        img1ori = cv2.equalizeHist(img1ori)
        img2ori = cv2.equalizeHist(img2ori)

        if i == 0:
            cv2.imwrite(TEMs_fine_dir + imgnamelist[i], img1ori)

        img1_Tpm = cv2.resize(img1ori, (
            int(img1ori.shape[1] / Graph_parameter.scale_factor),
            int(img1ori.shape[0] / Graph_parameter.scale_factor)),
                              interpolation=cv2.INTER_AREA)

        img2_Tpm = cv2.resize(img2ori, (
            int(img2ori.shape[1] / Graph_parameter.scale_factor),
            int(img2ori.shape[0] / Graph_parameter.scale_factor)),
                              interpolation=cv2.INTER_AREA)

        ##
        TMscore, TMlocb, TMlocc_reg, TMlocc_grid = SmallareaTemplatejMatch(img2_Tpm, img1_Tpm, Graph_parameter)

        ##
        ##
        BP_parameter = parser_BP.parse_args()
        BP_parameter.TMscore_thresholdH = np.median(TMscore)
        TMscore_loc_fixed = np.where(
            (TMscore > BP_parameter.TMscore_thresholdL) & (TMscore <= BP_parameter.TMscore_thresholdH))
        TMscore_loc_weight = np.where(TMscore > BP_parameter.TMscore_thresholdH)
        TMscore_loc_change = np.where(TMscore <= BP_parameter.TMscore_thresholdL)
        BP_parameter.TMscore_loc_fixed = TMscore_loc_fixed[0]
        BP_parameter.TMscore_loc_weight = TMscore_loc_weight[0]
        BP_parameter.TMscore_loc_change = TMscore_loc_change[0]  # flag = 1 要变的
        BP_parameter.graph_update_flag = np.ones(len(TMscore))
        BP_parameter.graph_update_flag[BP_parameter.TMscore_loc_fixed] = 0  # 0 不变，也不生成选择
        BP_parameter.graph_update_flag[BP_parameter.TMscore_loc_weight] = 2  # 2 不变，生成选择

        ##
        g, graph_dict = Elastic_Graph(Graph_parameter, TMscore, TMlocc_grid, TMlocc_reg, BP_parameter.graph_update_flag)

        ##
        even_out, odd_out = run_BP(TMscore, TMlocc_reg, TMlocc_grid, graph_dict, Graph_parameter, BP_parameter)

        ##
        img2_Tpmtemp = copy.deepcopy(img2_Tpm)
        for y in range(Graph_parameter.bigrow):
            for x in range(Graph_parameter.bigcol):
                cv2.rectangle(img2_Tpmtemp,
                              (int(even_out[y, x, 0]) - Graph_parameter.dst_radius,
                               int(even_out[y, x, 1]) - Graph_parameter.dst_radius),
                              (int(even_out[y, x, 0]) + Graph_parameter.dst_radius,
                               int(even_out[y, x, 1]) + Graph_parameter.dst_radius),
                              (0, 0, 0), 2)

        for y in range(Graph_parameter.bigrow - 1):
            for x in range(Graph_parameter.bigcol - 1):
                cv2.rectangle(img2_Tpmtemp,
                              (int(odd_out[y, x, 0]) - Graph_parameter.dst_radius,
                               int(odd_out[y, x, 1]) - Graph_parameter.dst_radius),
                              (int(odd_out[y, x, 0]) + Graph_parameter.dst_radius,
                               int(odd_out[y, x, 1]) + Graph_parameter.dst_radius),
                              (0, 0, 0), 2)
        cv2.imwrite(TEMs_reg_config_dir + 'BP_grid_' + imgnamelist[i + 1], img2_Tpmtemp)

        ##
        BPTMlocc_reg = []
        for numi in range(Graph_parameter.bigrow + (Graph_parameter.bigrow - 1)):
            rowi = int(numi / 2)
            if numi % 2 == 0:
                for numj in range(Graph_parameter.bigcol):
                    BPTMlocc_reg.append(even_out[rowi, numj])
            else:
                for numj in range(Graph_parameter.bigcol - 1):
                    BPTMlocc_reg.append(odd_out[rowi, numj])
        BPTMlocc_reg = np.array(BPTMlocc_reg)

        ##
        g, graph_dict = Elastic_Graph(Graph_parameter, TMscore, TMlocc_grid, BPTMlocc_reg,
                                      BP_parameter.graph_update_flag)
        ##
        dglg = dgl.from_networkx(g, node_attrs=['update_flag', 'nodescore', 'nodegrid', 'nodereg'])  # 转换成DGL图
        dglg.ndata['locdif'] = torch.zeros(dglg.num_nodes(), dtype=torch.int32)

        for epoch in range(Graph_parameter.Elasticiter):
            final_ft = updata_all_example(dglg)
            print(final_ft)

        final_loc = dglg.ndata['nodegrid']
        final_loc = torch.round(final_loc)
        final_loc = final_loc.numpy()

        ##
        N = len(TMlocc_grid)
        TMlocc_reg = TMlocc_reg * Graph_parameter.scale_factor
        TMlocc_grid = TMlocc_grid * Graph_parameter.scale_factor
        BP_loc = BPTMlocc_reg * Graph_parameter.scale_factor
        final_loc = final_loc * Graph_parameter.scale_factor

        TMlocc_grid = np.array(TMlocc_grid, np.int32)
        TMlocc_grid = TMlocc_grid.reshape(1, -1, 2)

        TMlocc_reg = np.array(TMlocc_reg, np.int32)
        TMlocc_reg = TMlocc_reg.reshape(1, -1, 2)

        BP_loc = np.array(BP_loc, np.int32)
        BP_loc = BP_loc.reshape(1, -1, 2)

        final_loc = np.array(final_loc, np.int32)
        final_loc = final_loc.reshape(1, -1, 2)

        matches = []
        for n in range(1, N + 1):
            matches.append(
                cv2.DMatch(n, n, 0))  # 查询点的索引（当前要寻找匹配结果的点在它所在图片上的索引） 被查询到的点的索引（存储库中的点的在存储库上的索引） 常为0(通常可理解为第几幅图)

        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.setRegularizationParameter(100000)

        tps.estimateTransformation(final_loc, TMlocc_grid, matches)  # 接口（模板特征点，目标特征点，匹配点）
        img = tps.warpImage(img2ori)
        cv2.imwrite(TEMs_fine_dir + imgnamelist[i + 1], img)

        if i == 0:
            imgTOM = cv2.imread(TOM_coarse_b_dir + imgnamelist[i])
            if imgTOM.ndim == 3:
                imgTOM = cv2.cvtColor(imgTOM, cv2.COLOR_RGB2GRAY)
            imgTOM = cv2.equalizeHist(imgTOM)
            cv2.imwrite(TOM_fine_dir + imgnamelist[i], imgTOM)

        imgTOM = cv2.imread(TOM_coarse_b_dir + imgnamelist[i + 1])
        if imgTOM.ndim == 3:
            imgTOM = cv2.cvtColor(imgTOM, cv2.COLOR_RGB2GRAY)
        imgTOM = cv2.equalizeHist(imgTOM)
        img = tps.warpImage(imgTOM)
        cv2.imwrite(TOM_fine_dir + imgnamelist[i + 1], img)

        scio.savemat(TEMs_reg_config_dir + 'TMlocc_grid' + str(i) + '.mat', {'TMlocc_grid': TMlocc_grid})
        scio.savemat(TEMs_reg_config_dir + 'final_loc' + str(i) + '.mat', {'final_loc': final_loc})