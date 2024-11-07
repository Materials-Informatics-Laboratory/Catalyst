from ...graph.graph import Atomic_Graph_Data, Generic_Graph_Data

import torch

def accumulate_predictions(pred,data,loss_tag):
    if loss_tag == 'exact':
        if hasattr(data, 'x_atm_batch'): #atomic graph
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
        if hasattr(data, 'node_G_batch'): # generic graph
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
        y = data.y.flatten()
    elif loss_tag == 'sum':
        preds = None
        for p in pred:
            if preds is None:
                preds = p.sum()
            else:
                preds += p.sum()
        y = data.y.flatten().sum()

    return preds, y




