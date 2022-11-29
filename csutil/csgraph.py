import cv2
import copy
import networkx as nx
import numpy as np

def SmallareaTemplatejMatch(img1, img2, TMparameter):
    tempimgdir = '/home/changs/ETregis/img/temp/'

    lefttopx = TMparameter.lefttopx
    lefttopy = TMparameter.lefttopy
    # rightbottomx = TMparameter.rightbottomx
    # rightbottomy = TMparameter.rightbottomy
    bigrow = TMparameter.bigrow  # n个点2n个区域
    bigcol = TMparameter.bigcol
    dst_radius = TMparameter.dst_radius
    search_radius = TMparameter.search_radius

    grid_radius_x = TMparameter.grid_radius_x
    grid_radius_y = TMparameter.grid_radius_y

    if (search_radius > grid_radius_x) | (search_radius > grid_radius_y):
        print('错误：搜索半径大于网格半径！')
    if (dst_radius >= search_radius):
        print('错误：目标半径大于等于搜索半径！')

    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    img1_search_grid = copy.deepcopy(img1)
    img1_TM_grid = copy.deepcopy(img1)
    img2_dst_grid = copy.deepcopy(img2)

    TMscore = []
    TMlocb = []
    TMlocc_grid = []
    TMlocc_reg = []
    for i in range(bigrow + (bigrow - 1)):  # bigrow + (bigrow - 1)    1的话只有一行  cv2.rectangle的点是先列后行
        if i % 2 == 0:
            for j in range(bigcol):
                grid_center_loc = (lefttopy + j * 2 * grid_radius_y + grid_radius_y, lefttopx + i * grid_radius_x + grid_radius_x)
                cv2.rectangle(img1_search_grid,
                              (grid_center_loc[0] - search_radius, grid_center_loc[1] - search_radius),
                              (grid_center_loc[0] + search_radius, grid_center_loc[1] + search_radius),
                              (0, 0, 225), 2)
                cv2.rectangle(img2_dst_grid,
                              (grid_center_loc[0] - dst_radius, grid_center_loc[1] - dst_radius),
                              (grid_center_loc[0] + dst_radius, grid_center_loc[1] + dst_radius),
                              (0, 0, 255), 2)
                imgtarget = copy.deepcopy(
                            img1[grid_center_loc[1] - search_radius: grid_center_loc[1] + search_radius,
                            grid_center_loc[0] - search_radius: grid_center_loc[0] + search_radius]
                            )
                imgtemplate = copy.deepcopy(
                            img2[grid_center_loc[1] - dst_radius: grid_center_loc[1] + dst_radius,
                            grid_center_loc[0] - dst_radius: grid_center_loc[0] + dst_radius]
                            )

                # cv2.imshow("imgtarget", imgtarget)
                # cv2.imshow("imgtemplate", imgtemplate)
                # cv2.waitKey()

                results = cv2.matchTemplate(imgtarget, imgtemplate, cv2.TM_CCOEFF_NORMED)   # cv2.TM_SQDIFF_NORMED利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results)

                min_loc = max_loc
                min_val = max_val

                # cv2.rectangle(imgtarget, min_loc, (min_loc[0] + dst_radius * 2, min_loc[1] + dst_radius * 2), (0, 0, 255), 2)
                min_locori = (min_loc[0] + grid_center_loc[0] - search_radius, min_loc[1] + grid_center_loc[1] - search_radius)
                cv2.rectangle(img1_TM_grid, min_locori, (min_locori[0] + dst_radius * 2, min_locori[1] + dst_radius * 2), (0, 0, 255), 2)

                min_locoric = (min_locori[0] + dst_radius, min_locori[1] + dst_radius)
                TMscore.append(min_val)
                TMlocb.append(min_locori)
                TMlocc_reg.append(min_locoric)
                TMlocc_grid.append(grid_center_loc)


                # cv2.imwrite(tempimgdir + str(i) + str(j) + 'target.jpg', imgtarget)
                # cv2.imwrite(tempimgdir + str(i) + str(j) + 'template.jpg', imgtemplate)
        else:
            for j in range(bigcol - 1):
                grid_center_loc = (lefttopy + j * 2 * grid_radius_y + grid_radius_y * 2, lefttopx + i * grid_radius_x + grid_radius_x)
                cv2.rectangle(img1_search_grid,
                              (grid_center_loc[0] - search_radius, grid_center_loc[1] - search_radius),
                              (grid_center_loc[0] + search_radius, grid_center_loc[1] + search_radius),
                              (0, 0, 225), 2)
                cv2.rectangle(img2_dst_grid,
                              (grid_center_loc[0] - dst_radius, grid_center_loc[1] - dst_radius),
                              (grid_center_loc[0] + dst_radius, grid_center_loc[1] + dst_radius),
                              (0, 0, 255), 2)
                imgtarget = copy.deepcopy(
                            img1[grid_center_loc[1] - search_radius: grid_center_loc[1] + search_radius,
                            grid_center_loc[0] - search_radius: grid_center_loc[0] + search_radius]
                            )
                imgtemplate = copy.deepcopy(
                            img2[grid_center_loc[1] - dst_radius: grid_center_loc[1] + dst_radius,
                            grid_center_loc[0] - dst_radius: grid_center_loc[0] + dst_radius]
                            )

                # cv2.imshow("imgtarget", imgtarget)
                # cv2.imshow("imgtemplate", imgtemplate)
                # cv2.waitKey()

                results = cv2.matchTemplate(imgtarget, imgtemplate,
                                            cv2.TM_CCOEFF_NORMED)  # cv2.TM_SQDIFF_NORMED利用平方差来进行匹配,最好匹配为0.匹配越差,匹配值越大
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results)

                min_loc = max_loc
                min_val = max_val

                # cv2.rectangle(imgtarget, min_loc, (min_loc[0] + dst_radius * 2, min_loc[1] + dst_radius * 2), (0, 0, 225), 2)
                min_locori = (min_loc[0] + grid_center_loc[0] - search_radius, min_loc[1] + grid_center_loc[1] - search_radius)
                cv2.rectangle(img1_TM_grid, min_locori, (min_locori[0] + dst_radius * 2, min_locori[1] + dst_radius * 2),
                              (0, 0, 255), 2)

                min_locoric = (min_locori[0] + dst_radius, min_locori[1] + dst_radius)
                TMscore.append(min_val)
                TMlocb.append(min_locori)
                TMlocc_reg.append(min_locoric)
                TMlocc_grid.append(grid_center_loc)

                # cv2.imwrite(tempimgdir + str(i) + str(j) + 'target.jpg', imgtarget)
                # cv2.imwrite(tempimgdir + str(i) + str(j) + 'template.jpg', imgtemplate)

    TMscore = np.array(TMscore)
    TMlocb = np.array(TMlocb)
    TMlocc_reg = np.array(TMlocc_reg)
    TMlocc_grid = np.array(TMlocc_grid)

    cv2.imwrite(tempimgdir + 'img1_search_grid.jpg', img1_search_grid)
    cv2.imwrite(tempimgdir + 'img1_TM_grid.jpg', img1_TM_grid)
    cv2.imwrite(tempimgdir + 'img2_dst_grid.jpg', img2_dst_grid)
    return TMscore, TMlocb, TMlocc_reg, TMlocc_grid

def Elastic_Graph(graph_param, TMscore, TMlocc_grid, TMlocc_reg, graph_update_flag):
    graph_row = graph_param.bigrow + graph_param.bigrow - 1
    g = nx.Graph()  # 创建空的无向图
    oneidx = 0
    graph_dict = {}
    ## 加点
    for i in range(graph_row):
        if i % 2 == 0:
            for j in range(graph_param.bigcol):
                g.add_node((i, j), update_flag=graph_update_flag[oneidx], nodescore=TMscore[oneidx],
                           nodegrid=TMlocc_grid[oneidx], nodereg=TMlocc_reg[oneidx])
                tempdic_item = {oneidx: (i, j)}
                graph_dict.update(tempdic_item)
                oneidx = oneidx + 1
        else:
            for j in range(graph_param.bigcol - 1):
                g.add_node((i, j), update_flag=graph_update_flag[oneidx], nodescore=TMscore[oneidx],
                           nodegrid=TMlocc_grid[oneidx], nodereg=TMlocc_reg[oneidx])
                tempdic_item = {oneidx: (i, j)}
                graph_dict.update(tempdic_item)
                oneidx = oneidx + 1

    ## 加边
    for i in range(graph_row):
        if i == graph_row - 1:
            for j in range(graph_param.bigcol - 1):
                g.add_edge((i, j), (i, j + 1), weight=1)
        elif i == graph_row - 2:
            for j in range(graph_param.bigcol - 1):
                if j == graph_param.bigcol - 2:
                    g.add_edge((i, j), (i + 1, j), weight=1)
                    g.add_edge((i, j), (i + 1, j + 1), weight=1)
                else:
                    g.add_edge((i, j), (i, j + 1), weight=1)
                    g.add_edge((i, j), (i + 1, j), weight=1)
                    g.add_edge((i, j), (i + 1, j + 1), weight=1)
        else:
            if i % 2 == 0:
                for j in range(graph_param.bigcol):
                    if j == 0:
                        g.add_edge((i, j), (i, j + 1), weight=1)
                        g.add_edge((i, j), (i + 1, j), weight=1)
                        g.add_edge((i, j), (i + 2, j), weight=1)
                    elif j == graph_param.bigcol - 1:
                        g.add_edge((i, j), (i + 1, j - 1), weight=1)
                        g.add_edge((i, j), (i + 2, j), weight=1)
                    else:
                        g.add_edge((i, j), (i, j + 1), weight=1)
                        g.add_edge((i, j), (i + 1, j - 1), weight=1)
                        g.add_edge((i, j), (i + 1, j), weight=1)
                        g.add_edge((i, j), (i + 2, j), weight=1)
            else:
                for j in range(graph_param.bigcol - 1):
                    if j == graph_param.bigcol - 2:
                        g.add_edge((i, j), (i + 1, j), weight=1)
                        g.add_edge((i, j), (i + 1, j + 1), weight=1)
                        g.add_edge((i, j), (i + 2, j), weight=1)
                    else:
                        g.add_edge((i, j), (i, j + 1), weight=1)
                        g.add_edge((i, j), (i + 1, j), weight=1)
                        g.add_edge((i, j), (i + 1, j + 1), weight=1)
                        g.add_edge((i, j), (i + 2, j), weight=1)
    return g, graph_dict