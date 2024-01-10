import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter
import os, sys
sys.path.insert(0, "/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT/benchmarking") 

import sys
sys.path.append("..") 

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torch_sparse import coalesce
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import to_networkx, to_undirected

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils import *
from get_heuristic import *
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
import pandas as pd 
from math import inf
import seaborn as sns
import matplotlib.pyplot  as plt 

def get_hist(A, full_A, num_nodes, data, use_heuristic, args):

    # Assuming you have a tensor with node indices
    nodes = torch.arange(num_nodes)
    # Generate pairwise combinations of indice
    scores = []
    num_cn = {}
    for src in tqdm(nodes):
        for dst in tqdm(nodes):
            cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
            if str(cur_scores[0]) in num_cn.keys():
                num_cn[str(cur_scores[0])] += 1
            else:
                num_cn[str(cur_scores[0])] = 1

    pos_test_pred = np.concatenate(scores, 0)
    np.savez_compressed(f'{data}_{use_heuristic}.npz', matrix=pos_test_pred)
    return pos_test_pred


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='CN')

    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--use_mrr', action='store_true', default=False)

    ####### katz
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--path_len', type=int, default=3)

    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--remove', action='store_true', default=False)
    

    args = parser.parse_args()
    print(args)

    # dataset = Planetoid('.', 'cora')
    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))
    
    data = dataset[0]
    
    use_heuristic = args.use_heuristic
    # split_edge = dataset.get_edge_split()
    node_num = data.num_nodes
    edge_index = data.edge_index

    if hasattr(data, 'edge_weight'):
        if data.edge_weight != None:
            # edge_weight = data.edge_weight.to(torch.float)
            edge_weight = data.edge_weight.view(-1).to(torch.float)
        else:
            edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
    else:
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

        idx = torch.tensor([1,0])
        edge_index = torch.cat([edge_index, edge_index[idx]], dim=1)
        edge_weight = torch.ones(edge_index.size(1), dtype=int)

    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                       shape=(node_num, node_num))
    
    if args.use_valedges_as_input:
        print('use validation!!!')
        val_edge_index = pos_valid_edge
        val_edge_index = to_undirected(val_edge_index)

        edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        val_edge_weight = torch.ones([val_edge_index.size(1)], dtype=int)

        edge_weight = torch.cat([edge_weight, val_edge_weight], 0)
        
        full_A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                        shape=(node_num, node_num)) 
    else:
        print('no validation!!!')

        full_A = A
    
    print('A: ', A.nnz)
    print('full_A: ', full_A.nnz)

    pos_test_pred = get_hist(A, full_A, node_num, use_heuristic, data, args)




if __name__ == "__main__":
   

    main()