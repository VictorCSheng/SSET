import numpy as np
import dgl
import torch

## https://blog.csdn.net/znsoft/article/details/114515868
def x_message_func(edges):   #  0.8
    msg = (edges.dst['nodegrid'] - edges.src['nodegrid']) - (edges.dst['nodereg'] - edges.src['nodegrid'])    ## h*norm   更新边权重 源头点的h*源头点的正则化参数
    msg = np.clip(msg, -2, 2)     ##
    return {'m': msg}

def x_gcn_reduce(nodes):  #
    dx = torch.sum(nodes.mailbox['m'], 1)  ##
    locdif = 0.9 * (nodes.data['nodereg'] - nodes.data['nodegrid']) + 0.2 * dx
    # nodes.data['locdif'] = locdif
    # nodes.data['nodereg'] = nodes.data['nodereg'] + locdif
    # nodereg = nodes.data['nodereg'] + dx
    # nodes.data['nodeloc'] = nodes.data['nodeloc'] + dx
    return {'locdif': locdif}


def updata_all_example(graph):
    graph.update_all(x_message_func,x_gcn_reduce)
    graph.ndata['nodegrid'] = graph.ndata['nodegrid'] + graph.ndata['locdif']
    elastic_erro = torch.sum(graph.ndata['locdif'])
    # graph.ndata['nodeloc'][1] = graph.ndata['nodeloc'][1] + graph.ndata['dx']
    return elastic_erro