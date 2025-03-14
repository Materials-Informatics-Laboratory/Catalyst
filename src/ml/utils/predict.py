from ...graph.graph import Atomic_Graph_Data, Generic_Graph_Data

import torch

import torch.distributed as dist
def accumulate_predictions(pred,data,loss_tag,return_y=True):
    if loss_tag == 'ensemble':
        pass
    else:
        if len(pred[0][0]) > 1:
            if loss_tag == 'exact':
                if hasattr(data, 'x_atm_batch'): #atomic graph
                    x = torch.unique(data['x_atm_batch'])
                    sorted_nodes_G = [[None for j in range(len(pred[0][0]))] for i in range(len(x))]
                    dd = data['x_atm_batch']
                    for i, xx in enumerate(x):
                        mask = torch.where(dd == xx, torch.tensor(1), torch.tensor(0))
                        indices = torch.nonzero(mask).squeeze()
                        for j in range(len(pred[0][0])):
                            sorted_nodes_G[i][j] = pred[0][indices, j]
                    x = torch.unique(data['x_bnd_batch'])
                    sorted_nodes_A = [[None for j in range(len(pred[1][0]))] for i in range(len(x))]
                    dd = data['x_bnd_batch']
                    for i, xx in enumerate(x):
                        mask = torch.where(dd == xx, torch.tensor(1), torch.tensor(0))
                        indices = torch.nonzero(mask).squeeze()
                        for j in range(len(pred[1][0])):
                            sorted_nodes_A[i][j] = pred[1][indices, j]
                    if hasattr(data, 'x_ang_batch'):
                        x = torch.unique(data['x_ang_batch'])
                        sorted_edges_A = [[None for j in range(len(pred[2][0]))] for i in range(len(x))]
                        dd = data['x_ang_batch']
                        for i, xx in enumerate(x):
                            mask = torch.where(dd == xx, torch.tensor(1), torch.tensor(0))
                            indices = torch.nonzero(mask).squeeze()
                            for j in range(len(pred[2][0])):
                                sorted_edges_A[i][j] = pred[2][indices, j]
                    preds = []
                    for i in range(len(sorted_nodes_G)):
                        py = torch.zeros(len(sorted_nodes_G[i]))
                        if hasattr(data, 'x_ang_batch'):
                            for j in range(len(sorted_nodes_G[i])):
                                py[j] = sorted_nodes_G[i][j].sum() + sorted_nodes_A[i][j].sum() + sorted_edges_A[i][j].sum()
                        else:
                            for j in range(len(sorted_nodes_G[i])):
                                py[j] = sorted_nodes_G[i][j].sum() + sorted_nodes_A[i][j].sum()
                        preds.append(py)
                if hasattr(data, 'node_G_batch'): # generic graph
                    x = torch.unique(data['node_G_batch'])
                    sorted_nodes_G = [[None for j in range(len(pred[0][0]))] for i in range(len(x))]
                    dd = data['node_G_batch']
                    for i, xx in enumerate(x):
                        mask = torch.where(dd == xx,torch.tensor(1), torch.tensor(0))
                        indices = torch.nonzero(mask).squeeze()
                        for j in range(len(pred[0][0])):
                            sorted_nodes_G[i][j] = pred[0][indices,j]
                    x = torch.unique(data['node_A_batch'])
                    sorted_nodes_A = [[None for j in range(len(pred[1][0]))] for i in range(len(x))]
                    dd = data['node_A_batch']
                    for i, xx in enumerate(x):
                        mask = torch.where(dd == xx, torch.tensor(1), torch.tensor(0))
                        indices = torch.nonzero(mask).squeeze()
                        for j in range(len(pred[1][0])):
                            sorted_nodes_A[i][j] = pred[1][indices, j]
                    if hasattr(data, 'edge_A_batch'):
                        x = torch.unique(data['edge_A_batch'])
                        sorted_edges_A = [[None for j in range(len(pred[2][0]))] for i in range(len(x))]
                        dd = data['edge_A_batch']
                        for i, xx in enumerate(x):
                            mask = torch.where(dd == xx, torch.tensor(1), torch.tensor(0))
                            indices = torch.nonzero(mask).squeeze()
                            for j in range(len(pred[2][0])):
                                sorted_edges_A[i][j] = pred[2][indices, j]
                    preds = []
                    for i in range(len(sorted_nodes_G)):
                        py = torch.zeros(len(sorted_nodes_G[i]))
                        if hasattr(data, 'edge_A_batch'):
                            for j in range(len(sorted_nodes_G[i])):
                                py[j] = sorted_nodes_G[i][j].sum() + sorted_nodes_A[i][j].sum() + sorted_edges_A[i][j].sum()
                        else:
                            for j in range(len(sorted_nodes_G[i])):
                                py[j] = sorted_nodes_G[i][j].sum() + sorted_nodes_A[i][j].sum()
                        preds.append(py)
                preds = torch.stack(preds, dim=1)
                if return_y:
                    y = torch.stack(data.y)

            elif loss_tag == 'sum':
                preds = [0.0]*len(pred[0][0])
                for i in range(len(pred)):
                    py = torch.stack(tuple(pred[i]),dim=1)
                    preds[i] = torch.sum(py)
                if return_y:
                    y = [None] * len(data.y)
                    for i in range(len(data.y)):
                         y[i] = data.y[i].sum()
        else:
            if loss_tag == 'exact':
                if hasattr(data, 'x_atm_batch'):  # atomic graph
                    d = pred[0].flatten()
                    x = torch.unique(data['x_atm_batch'])
                    sorted_atms = [None] * len(x)
                    dd = data['x_atm_batch']
                    for i, xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_atms[i] = d[xw]
                    d = pred[1].flatten()
                    x = torch.unique(data['x_bnd_batch'])
                    sorted_bnds = [None] * len(x)
                    dd = data['x_bnd_batch']
                    for i, xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_bnds[i] = d[xw]
                    if hasattr(data, 'x_ang_batch'):
                        d = pred[2].flatten()
                        x = torch.unique(data['x_ang_batch'])
                        sorted_angs = [None] * len(x)
                        dd = data['x_ang_batch']
                        for i, xx in enumerate(x):
                            xw = torch.where(dd == xx)
                            sorted_angs[i] = d[xw]
                    preds = []
                    for i in range(len(sorted_atms)):
                        if hasattr(data, 'x_ang_batch'):
                            preds.append(sorted_atms[i].sum() + sorted_angs[i].sum() + sorted_bnds[i].sum())
                        else:
                            preds.append(sorted_atms[i].sum() + sorted_bnds[i].sum())
                    preds = torch.stack(preds)
                if hasattr(data, 'node_G_batch'):  # generic graph
                    d = pred[0].flatten()
                    x = torch.unique(data['node_G_batch'])
                    sorted_nodes_G = [None] * len(x)
                    dd = data['node_G_batch']
                    for i, xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_nodes_G[i] = d[xw]
                    d = pred[1].flatten()
                    x = torch.unique(data['node_A_batch'])
                    sorted_nodes_A = [None] * len(x)
                    dd = data['node_A_batch']
                    for i, xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_nodes_A[i] = d[xw]
                    if hasattr(data, 'edge_A_batch'):
                        d = pred[2].flatten()
                        x = torch.unique(data['edge_A_batch'])
                        sorted_edges_A = [None] * len(x)
                        dd = data['edge_A_batch']
                        for i, xx in enumerate(x):
                            xw = torch.where(dd == xx)
                            sorted_edges_A[i] = d[xw]
                    preds = []
                    for i in range(len(sorted_nodes_G)):
                        if hasattr(data, 'edge_A_batch'):
                            preds.append(sorted_nodes_G[i].sum() + sorted_nodes_A[i].sum() + sorted_edges_A[i].sum())
                        else:
                            preds.append(sorted_nodes_G[i].sum() + sorted_nodes_A[i].sum())
                    preds = torch.stack(preds)
                if return_y:
                    if hasattr(data, 'y'):
                        if len(data.y) == 1:
                            y = data.y[0].flatten()
                        else:
                            y = data.y.flatten()
                    else:
                        y = None
            elif loss_tag == 'sum':
                preds = None
                for p in pred:
                    if preds is None:
                        preds = p.sum()
                    else:
                        preds += p.sum()
                if return_y:
                    if hasattr(data, 'y'):
                        if len(data.y) == 1:
                            y = data.y[0].flatten().sum()
                        else:
                            y = data.y.flatten().sum()
                    else:
                        None
        if return_y:
            if len(pred[0][0]) > 1:
                return torch.stack(tuple(preds)), torch.stack(tuple(y)), True
            else:
                return preds, y, False
        else:
            if len(pred[0][0]) > 1:
                return torch.stack(tuple(preds)), True
            else:
                return preds, False




