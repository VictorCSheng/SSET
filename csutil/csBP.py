import scipy.io as scio
import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import numpy as np
import copy

from csutil.csgraph import Elastic_Graph

##
def dt(h,lgdif_locini,lgdif_locnei):
    n = len(h)

    dtemp = np.zeros(n)       #
    v = np.zeros(n)       #
    z = np.zeros(n + 1)   #

    v[0] = 0
    z[0] = float('-inf')
    z[1] = float('inf')

    lgdif_locnei_sortid = np.argsort(lgdif_locnei)
    lgdif_locnei_temp = lgdif_locnei[lgdif_locnei_sortid]
    htemp = h[lgdif_locnei_sortid]

    ##
    j = 0  # j-1
    for q in range(1,n-1,1):
        s = ((htemp[q] + (lgdif_locnei_temp[q]) ** 2) - (htemp[int(v[j])] + (lgdif_locnei_temp[int(v[j])]) ** 2)) \
            / (2 * (lgdif_locnei_temp[q] - lgdif_locnei_temp[int(v[j])]) + 0.0000001)
        while (s <= z[j]):
            j = j - 1
            s = ((htemp[q] + (lgdif_locnei_temp[q]) ** 2) - (htemp[int(v[j])] + (lgdif_locnei_temp[int(v[j])]) ** 2)) \
                / (2 * (lgdif_locnei_temp[q] - lgdif_locnei_temp[int(v[j])]) + 0.0000001)

        j = j + 1
        v[j] = q    #
        z[j] = s    #
        z[j + 1] = float('inf')

    ##
    j = 0
    lgdif_locini_sortid = np.argsort(lgdif_locini)
    lgdif_locini_temp = lgdif_locini[lgdif_locini_sortid]
    for q in range(n):
        while (z[j] < lgdif_locini_temp[q]):
            j = j + 1
        dtemp[q] = (lgdif_locini_temp[q]-lgdif_locnei_temp[int(v[j])])**2 + htemp[int(v[j])]

    d = np.zeros(n)
    d[lgdif_locini_sortid] = dtemp
    return d

##
def msg(direction1, direction2, direction3, datacost, lgdif_locini, lgdif_locnei, BPparameter):
    #
    dst = direction1 + direction2 + direction3 + datacost   # 要往某个方向传播的消息的汇总，三个方向加一个自身datacost
    miniC = min(dst)
    tmp = dt(dst,lgdif_locini,lgdif_locnei)

    #
    disc_k = BPparameter.disc_k
    miniC = miniC + disc_k
    dst = [min(x, miniC) for x in tmp]
    dst = np.array(dst)

    #
    val = sum(dst[:])
    val = val / (len(lgdif_locini))
    dst = dst - val

    return dst

def bp_cp(u, d, l, r, datacost, lgdif_loc, BPparameter):
    u = np.pad(u, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    d = np.pad(d, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    l = np.pad(l, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    r = np.pad(r, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    datacost = np.pad(datacost, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    lgdif_loc = np.pad(lgdif_loc, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=(0, 0))
    height_or_weidth = np.size(datacost, 0)

    numIter = BPparameter.numIter

    for iter in range(numIter):
        print('Iteration: ' + str(iter))
        for m in range(1, (height_or_weidth - 1)):
            for n in range((m + iter) % 2 + 1, (height_or_weidth - 1), 2):
                u[m, n, :] = msg(u[m+1, n, :], l[m, n+1, :], r[m, n-1, :], datacost[m,n,:], lgdif_loc[m,n,:], lgdif_loc[m+1, n, :], BPparameter)        # msg(u(y+1,x,:), l(y,x+1,:), r(y,x-1,:), data(y,x,:));
                d[m, n, :] = msg(d[m-1, n, :], l[m, n+1, :], r[m, n-1, :], datacost[m,n,:], lgdif_loc[m,n,:], lgdif_loc[m-1, n, :], BPparameter)        # msg(d(y-1,x,:), l(y,x+1,:), r(y,x-1,:), data(y,x,:));
                r[m, n, :] = msg(u[m+1, n, :], d[m-1, n, :], r[m, n-1, :], datacost[m,n,:], lgdif_loc[m,n,:], lgdif_loc[m, n-1, :], BPparameter)        # msg(u(y+1,x,:), d(y-1,x,:), r(y,x-1,:), data(y,x,:));
                l[m, n, :] = msg(u[m+1, n, :], d[m-1, n, :], l[m, n+1, :], datacost[m,n,:], lgdif_loc[m,n,:], lgdif_loc[m, n+1, :], BPparameter)        # msg(u(y+1,x,:), d(y-1,x,:), l(y,x+1,:), data(y,x,:));

    return u,d,l,r

##
def run_BP(TMscore, TMlocc_reg, TMlocc_grid, graph_dict, Graphparameter, BPparameter):
    TMscore_loc_weight = BPparameter.TMscore_loc_weight
    TMscore_loc_change = BPparameter.TMscore_loc_change

    ##
    ############
    ##
    bplambda = BPparameter.bplambda       ## 0.05
    data_k = BPparameter.data_k           ## 10  ############
    even_dataCost_x = np.zeros((Graphparameter.bigrow, Graphparameter.bigcol, len(TMscore_loc_weight)))
    even_dataCost_y = np.zeros((Graphparameter.bigrow, Graphparameter.bigcol, len(TMscore_loc_weight)))
    odd_dataCost_x = np.zeros((Graphparameter.bigrow - 1, Graphparameter.bigcol - 1, len(TMscore_loc_weight)))
    odd_dataCost_y = np.zeros((Graphparameter.bigrow - 1, Graphparameter.bigcol - 1, len(TMscore_loc_weight)))

    even_lgdif_loc = np.zeros((Graphparameter.bigrow, Graphparameter.bigcol, len(TMscore_loc_weight), 2))
    odd_lgdif_loc = np.zeros((Graphparameter.bigrow - 1, Graphparameter.bigcol - 1, len(TMscore_loc_weight), 2))

    even_lable_loc = np.zeros((Graphparameter.bigrow, Graphparameter.bigcol, len(TMscore_loc_weight), 2))
    odd_lable_loc = np.zeros((Graphparameter.bigrow - 1, Graphparameter.bigcol - 1, len(TMscore_loc_weight), 2))
    all_lable_loc = np.zeros((len(TMscore), len(TMscore_loc_weight), 2))

    even_graph_dic = {}  #
    odd_graph_dic = {}  #
    for i in range(len(TMscore)):
        if i in TMscore_loc_change:
            change_id = i
            change_graph_id = graph_dict[change_id]

            for j in range(len(TMscore_loc_weight)):
                weight_id = TMscore_loc_weight[j]
                weight_graph_id = graph_dict[weight_id]
                newlable_row = TMlocc_reg[weight_id][1] - (weight_graph_id[0] - change_graph_id[0]) * Graphparameter.grid_radius_x
                if ((weight_graph_id[0] - change_graph_id[0]) % 2) == 0:
                    newlable_col = TMlocc_reg[weight_id][0] - (
                                weight_graph_id[1] - change_graph_id[1]) * Graphparameter.grid_radius_y * 2
                else:
                    if (change_graph_id[0] % 2) == 0:
                        newlable_col = TMlocc_reg[weight_id][0] - (
                                    weight_graph_id[1] - change_graph_id[1]) * Graphparameter.grid_radius_y * 2 - Graphparameter.grid_radius_y
                    else:
                        newlable_col = TMlocc_reg[weight_id][0] - (
                                    weight_graph_id[1] - change_graph_id[1]) * Graphparameter.grid_radius_y * 2 + Graphparameter.grid_radius_y

                all_lable_loc[change_id, j, :] = [newlable_col, newlable_row]
                ##(TMlocc_reg[change_id][0] - newlable_col) ** 2 + (TMlocc_reg[change_id][1] - newlable_row) ** 2
                gg_y = (TMlocc_reg[change_id][0] - newlable_col) ** 2
                gg_x = (TMlocc_reg[change_id][1] - newlable_row) ** 2
                gg_y = min(gg_y, data_k)
                gg_x = min(gg_x, data_k)

                lgdif_y = newlable_col - TMlocc_grid[change_id][0]
                lgdif_x = newlable_row - TMlocc_grid[change_id][1]
                lgdif_loc = [lgdif_y, lgdif_x]

                if (change_graph_id[0] % 2) == 0:
                    tempdic_item = {change_id: change_graph_id}
                    even_graph_dic.update(tempdic_item)

                    even_dataCost_y[int(change_graph_id[0] / 2), change_graph_id[1], j] = bplambda * gg_y
                    even_dataCost_x[int(change_graph_id[0] / 2), change_graph_id[1], j] = bplambda * gg_x

                    even_lgdif_loc[int(change_graph_id[0] / 2), change_graph_id[1], j, :] = lgdif_loc
                    even_lable_loc[int(change_graph_id[0] / 2), change_graph_id[1], j, :] = [newlable_col, newlable_row]
                else:
                    tempdic_item = {change_id: change_graph_id}
                    odd_graph_dic.update(tempdic_item)

                    odd_dataCost_y[int((change_graph_id[0] - 1) / 2), change_graph_id[1], j] = bplambda * gg_y
                    odd_dataCost_x[int((change_graph_id[0] - 1) / 2), change_graph_id[1], j] = bplambda * gg_x

                    odd_lgdif_loc[int((change_graph_id[0] - 1) / 2), change_graph_id[1], j, :] = lgdif_loc
                    odd_lable_loc[int((change_graph_id[0] - 1) / 2), change_graph_id[1], j, :] = [newlable_col,
                                                                                                  newlable_row]
        else:
            lgdif_y = TMlocc_reg[i][0] - TMlocc_grid[i][0]
            lgdif_x = TMlocc_reg[i][1] - TMlocc_grid[i][1]
            lgdif_loc = [lgdif_y, lgdif_x]
            lgdif_loc = np.tile(lgdif_loc, (len(TMscore_loc_weight), 1))

            fixed_graph_id = graph_dict[i]
            if (fixed_graph_id[0] % 2) == 0:
                tempdic_item = {i: fixed_graph_id}
                even_graph_dic.update(tempdic_item)
                even_lgdif_loc[int(fixed_graph_id[0] / 2), fixed_graph_id[1], :, :] = lgdif_loc
                even_lable_loc[int(fixed_graph_id[0] / 2), fixed_graph_id[1], :, :] = np.tile(TMlocc_reg[i], (
                len(TMscore_loc_weight), 1))
            else:
                tempdic_item = {i: fixed_graph_id}
                odd_graph_dic.update(tempdic_item)
                odd_lgdif_loc[int((fixed_graph_id[0] - 1) / 2), fixed_graph_id[1], :, :] = lgdif_loc
                odd_lable_loc[int((fixed_graph_id[0] - 1) / 2), fixed_graph_id[1], :, :] = np.tile(TMlocc_reg[i], (
                len(TMscore_loc_weight), 1))

    #############
    ####
    ## y
    u = copy.deepcopy(even_dataCost_y)
    d = copy.deepcopy(even_dataCost_y)
    l = copy.deepcopy(even_dataCost_y)
    r = copy.deepcopy(even_dataCost_y)
    u, d, l, r = bp_cp(u, d, l, r, even_dataCost_y, even_lgdif_loc[:, :, :, 0], BPparameter)
    ##
    even_out = np.zeros((Graphparameter.bigrow, Graphparameter.bigcol, 2))
    for y in range(1, Graphparameter.bigrow + 1, 1):
        for x in range(1, Graphparameter.bigcol + 1, 1):
            v = u[y + 1, x, :] + d[y - 1, x, :] + l[y, x + 1, :] + r[y, x - 1, :] + even_dataCost_y[y - 1, x - 1]
            lable_id = np.argmin(v)
            even_out[y - 1, x - 1, 0] = even_lable_loc[y - 1, x - 1, lable_id, 0]

    ## x
    u = copy.deepcopy(even_dataCost_x)
    d = copy.deepcopy(even_dataCost_x)
    l = copy.deepcopy(even_dataCost_x)
    r = copy.deepcopy(even_dataCost_x)
    u, d, l, r = bp_cp(u, d, l, r, even_dataCost_x, even_lgdif_loc[:, :, :, 1], BPparameter)
    ##
    for y in range(1, Graphparameter.bigrow + 1, 1):
        for x in range(1, Graphparameter.bigcol + 1, 1):
            v = u[y + 1, x, :] + d[y - 1, x, :] + l[y, x + 1, :] + r[y, x - 1, :] + even_dataCost_x[y - 1, x - 1]
            lable_id = np.argmin(v)
            even_out[y - 1, x - 1, 1] = even_lable_loc[y - 1, x - 1, lable_id, 1]

    ####
    ## y
    u = copy.deepcopy(odd_dataCost_y)
    d = copy.deepcopy(odd_dataCost_y)
    l = copy.deepcopy(odd_dataCost_y)
    r = copy.deepcopy(odd_dataCost_y)
    u, d, l, r = bp_cp(u, d, l, r, odd_dataCost_y, odd_lgdif_loc[:, :, :, 0], BPparameter)
    ##
    odd_out = np.zeros((Graphparameter.bigrow - 1, Graphparameter.bigcol - 1, 2))
    for y in range(1, Graphparameter.bigrow, 1):
        for x in range(1, Graphparameter.bigcol, 1):
            v = u[y + 1, x, :] + d[y - 1, x, :] + l[y, x + 1, :] + r[y, x - 1, :] + odd_dataCost_y[y - 1, x - 1]
            lable_id = np.argmin(v)
            odd_out[y - 1, x - 1, 0] = odd_lable_loc[y - 1, x - 1, lable_id, 0]

    ## x
    u = copy.deepcopy(odd_dataCost_x)
    d = copy.deepcopy(odd_dataCost_x)
    l = copy.deepcopy(odd_dataCost_x)
    r = copy.deepcopy(odd_dataCost_x)
    u, d, l, r = bp_cp(u, d, l, r, odd_dataCost_x, odd_lgdif_loc[:, :, :, 1], BPparameter)
    ##
    for y in range(1, Graphparameter.bigrow, 1):
        for x in range(1, Graphparameter.bigcol, 1):
            v = u[y + 1, x, :] + d[y - 1, x, :] + l[y, x + 1, :] + r[y, x - 1, :] + odd_dataCost_x[y - 1, x - 1]
            lable_id = np.argmin(v)
            odd_out[y - 1, x - 1, 1] = odd_lable_loc[y - 1, x - 1, lable_id, 1]

    return even_out, odd_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmallareaTemplatejMatch parameter')
    parser.add_argument('--lefttopx', default=0, type=int, help='lefttopx')
    parser.add_argument('--lefttopy', default=90, type=int, help='lefttopy')  # 90  11
    parser.add_argument('--rightbottomx', default=4095, type=int, help='rightbottomx')  # 4095  512
    parser.add_argument('--rightbottomy', default=4095, type=int, help='rightbottomy')  # 4095
    parser.add_argument('--bigrow', default=5, type=int, help='bigrow')
    parser.add_argument('--bigcol', default=5, type=int, help='bigcol')
    parser.add_argument('--dst_radius', default=35, type=int, help='dst_radius')
    parser.add_argument('--search_radius', default=45, type=int, help='dst_radius')
    parser.add_argument('--scale_factor', default=8, type=int, help='scale_factor')

    opt = parser.parse_args()

    opt.lefttopx = int(opt.lefttopx / opt.scale_factor)
    opt.lefttopy = int(opt.lefttopy / opt.scale_factor)
    opt.rightbottomx = int(opt.rightbottomx / opt.scale_factor)
    opt.rightbottomy = int(opt.rightbottomy / opt.scale_factor)

    opt.grid_radius_x = (opt.rightbottomx - opt.lefttopx) // (2 * opt.bigrow)
    opt.grid_radius_y = (opt.rightbottomy - opt.lefttopy) // (2 * opt.bigcol)

    tempdatadir = '../img/tempdata/'
    data = scio.loadmat(tempdatadir + 'TMscore.mat')
    TMscore = data['TMscore']
    TMscore = TMscore[0]
    data = scio.loadmat(tempdatadir + 'TMlocc_reg.mat')
    TMlocc_reg = data['TMlocc_reg']
    data = scio.loadmat(tempdatadir + 'TMlocc_grid.mat')
    TMlocc_grid = data['TMlocc_grid']

    ##
    TMscore_thresholdL = 0.15
    TMscore_thresholdH = np.median(TMscore)
    TMscore_loc_fixed = np.where((TMscore > TMscore_thresholdL) & (TMscore <= TMscore_thresholdH))
    TMscore_loc_weight = np.where(TMscore > TMscore_thresholdH)
    TMscore_loc_change = np.where(TMscore <= TMscore_thresholdL)
    TMscore_loc_fixed = TMscore_loc_fixed[0]
    TMscore_loc_weight = TMscore_loc_weight[0]
    TMscore_loc_change = TMscore_loc_change[0]   #
    graph_update_flag = np.ones(len(TMscore))
    graph_update_flag[TMscore_loc_fixed] = 0    #
    graph_update_flag[TMscore_loc_weight] = 2   #

    g, graph_dict = Elastic_Graph(opt, TMscore, TMlocc_reg, graph_update_flag)

    parser_BP = argparse.ArgumentParser(description='BP parameter')
    parser_BP.add_argument('--TMscore_thresholdL', default=0.15, type=int, help='TMscore_thresholdL')
    parser_BP.add_argument('--numIter', default=8, type=int, help='numIter')  ##BPparameter.numIter = 10
    parser_BP.add_argument('--bplambda', default=0.1, type=int, help='bplambda')    ### 0.05 datacost所占的比重
    parser_BP.add_argument('--data_k', default=200, type=int, help='data_k')  ### 10  ############datacost截断阈值
    parser_BP.add_argument('--disc_k', default=50, type=int, help='disc_k') ### 200  ############信息传递中的截断阈值
    BPparameter = parser_BP.parse_args()
    even_out, odd_out = run_BP(TMscore, TMlocc_reg, TMlocc_grid, graph_dict, opt, BPparameter)

    import cv2
    coarse_dir = '/home/changs/ETregis/img/coarse/'  # 存放对齐的可视化结果
    img1ori = cv2.imread(coarse_dir + '1.tif')
    img1ori = cv2.resize(img1ori, (int(img1ori.shape[1] / opt.scale_factor), int(img1ori.shape[0] / opt.scale_factor)),
                         interpolation=cv2.INTER_AREA)
    if img1ori.ndim == 3:
        imgtemp1 = cv2.cvtColor(img1ori, cv2.COLOR_RGB2GRAY)
    #
    imgtemp1 = cv2.equalizeHist(imgtemp1)
    for y in range(opt.bigrow):
        for x in range(opt.bigcol):
            cv2.rectangle(imgtemp1,
                          (int(even_out[y, x, 0]) - opt.dst_radius, int(even_out[y, x, 1]) - opt.dst_radius),
                          (int(even_out[y, x, 0]) + opt.dst_radius, int(even_out[y, x, 1]) + opt.dst_radius),
                          (0, 0, 0), 2)

    for y in range(opt.bigrow - 1):
        for x in range(opt.bigcol - 1):
            cv2.rectangle(imgtemp1,
                          (int(odd_out[y, x, 0]) - opt.dst_radius, int(odd_out[y, x, 1]) - opt.dst_radius),
                          (int(odd_out[y, x, 0]) + opt.dst_radius, int(odd_out[y, x, 1]) + opt.dst_radius),
                          (0, 0, 0), 2)

    cv2.imwrite(coarse_dir + '1temp.jpg', imgtemp1)

    tempa = 1