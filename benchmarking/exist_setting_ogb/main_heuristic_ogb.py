import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from collections import Counter
import os, sys
sys.path.insert(0, "/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/benchmarking") 

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
from pdb import set_trace as bp

def get_prediction(A, full_A, use_heuristic, pos_val_edge, neg_val_edge, 
                   pos_test_edge, neg_test_edge, pos_train_edge,
                   args):

    beta, path_len = args.beta, args.path_len
    remove = args.remove

    if 'katz' in use_heuristic:
        pos_val_pred = eval(use_heuristic)( A, pos_val_edge, beta, path_len, remove)
        neg_val_pred = eval(use_heuristic)( A, neg_val_edge, beta, path_len, remove)

        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, beta, path_len, remove)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, beta, path_len, remove)

    elif use_heuristic == 'shortest_path':
        pos_val_pred = eval(use_heuristic)( A, pos_val_edge, remove)
        neg_val_pred = eval(use_heuristic)( A, neg_val_edge, remove)

        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge, remove)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge, remove)

    else:  
        print('evaluate pos vlaid: ')
        pos_val_pred = eval(use_heuristic)(A, pos_val_edge)
        print('evaluate neg vlaid: ')
        neg_val_pred = eval(use_heuristic)(A, neg_val_edge)
        
        pos_train_pred = eval(use_heuristic)(A, pos_train_edge)
        
        print('evaluate pos test: ')
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_edge)
        print('evaluate neg test: ')

        neg_test_pred = eval(use_heuristic)(full_A, neg_test_edge)

    return pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, pos_train_pred

import numpy as np
from scipy.spatial import distance

def get_fp_prediction(data, test_pos, test_neg, args):
    test_pos, test_neg = test_pos.numpy().transpose(), test_neg.numpy().transpose()
    test_pos_pred, test_neg_pred = [], []

    distance_metric = {
        'cos': distance.cosine,
        'l2': distance.euclidean,
        'hamming': distance.hamming,
        'jaccard': distance.jaccard,
        'dice': distance.dice,
        'dot': lambda x, y: np.dot(x, y)
    }

    if args.distance not in distance_metric:
        raise ValueError("Invalid distance metric specified.")

    metric_function = distance_metric[args.distance]

    for ind in test_pos:
        metric_value = metric_function(data[ind[0]], data[ind[1]])
        test_pos_pred.append(metric_value)

    for ind_n in test_neg:
        metric_value = metric_function(data[ind_n[0]], data[ind_n[1]])
        test_neg_pred.append(metric_value)

    test_pos_pred, test_neg_pred = torch.tensor(np.asarray(test_pos_pred)), torch.tensor(np.asarray(test_neg_pred))
    return test_pos_pred, test_neg_pred



def get_metric_citation2(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_val_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, use_mrr):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    k_list  = [20, 50, 100]
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)
    
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    
    result = {}

    result_hit = {}
    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    if use_mrr:
        result_mrr_test = evaluate_mrr( pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1))
        
        result_mrr_val = evaluate_mrr( pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1))
        
        result_mrr = {}

    
        result['MRR'] = (result_mrr_val['MRR'], result_mrr_test['MRR'])
        result['mrr_hit20']  = (result_mrr_val['mrr_hit20'], result_mrr_test['mrr_hit20'])
        result['mrr_hit50']  = (result_mrr_val['mrr_hit50'], result_mrr_test['mrr_hit50'])
        
        result['mrr_hit100']  = (result_mrr_val['mrr_hit100'], result_mrr_test['mrr_hit100'])
    

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    result_auc = {}
    result['AUC'] = (result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_val['AP'], result_auc_test['AP'])


    return result

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def get_hist(args, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,  pos_train_pred):
    cn_dist = torch.cat((pos_val_pred, 
                         neg_val_pred, 
                         pos_test_pred,
                         neg_test_pred,
                         pos_train_pred), 0).numpy()
    data_df = pd.DataFrame({'size': cn_dist})

    data_df_filtered = data_df[data_df['size'] != 0.0]
    
    
def get_test_hist(args, A, pos_test_pred, neg_test_pred, use_heuristic):
   
    bin_edges = [-1, 0, 1, 3, 10, 25, float('inf')]
    
    pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    hist, bin_edges = np.histogram(pred, bins=bin_edges)
    
    hist = hist / hist.sum()
    plt.figure(figsize=(10, 8))
    plt.bar([1, 2, 3, 4, 5], hist)
    
    custom_ticks = [1, 2, 3, 4, 5]
    custom_labels = ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf']

    dirpath = '/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/benchmarking/exist_setting_ogb'
    
    plt.xticks(custom_ticks, custom_labels)
    plt.title(f'{args.data_name}_{use_heuristic}_filtered')
    plt.xlabel('Num of CN')  
    plt.ylabel('Propotion')  
    plt.title(f'CN distribution of {args.data_name}')
    plt.savefig(f'{dirpath}/{args.data_name}_{use_heuristic}_test_filtered.png')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=[1, 2, 3, 4, 5], y=hist, color='skyblue')
    plt.xticks([0, 1, 2, 3, 4], ['[0-1]', '[1-3]', '[3-10]', '[10-25]', '25-inf'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('CN distribution', fontsize=24)
    plt.xlabel('Num of CN', fontsize=20)
    plt.ylabel('Proportion', fontsize=20)
    plt.savefig(f'{dirpath}/sns{args.data_name}_{use_heuristic}_test_filtered.png')
        

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--use_heuristic', type=str, default='FP')

    parser.add_argument('--use_valedges_as_input', action='store_true', default=False)
    parser.add_argument('--use_mrr', action='store_true', default=False)

    ####### katz
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--path_len', type=int, default=3)

    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--remove', action='store_true', default=False)
    parser.add_argument('--distance', type=str, default='dot')
    args = parser.parse_args()
    print(args)

    # dataset = Planetoid('.', 'cora')
    dataset = PygLinkPropPredDataset(name=args.data_name, root=os.path.join(get_root_dir(), "dataset", args.data_name))

    data = dataset[0]
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(g)
    
    use_heuristic = args.use_heuristic
    split_edge = dataset.get_edge_split()
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

    if args.data_name != 'ogbl-citation2':
        pos_train_edge = split_edge['train']['edge'].t()
        pos_valid_edge = split_edge['valid']['edge'].t()
        neg_valid_edge = split_edge['valid']['edge_neg'].t()
        pos_test_edge = split_edge['test']['edge'].t()
        neg_test_edge = split_edge['test']['edge_neg'].t()
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

        # idx = torch.randperm(split_edge['train']['source_node'].numel())[:split_edge['valid']['source_node'].size(0)]
        # source, target = split_edge['train']['source_node'][idx], split_edge['train']['target_node'][idx]
        # train_val_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        val_neg_edge = split_edge['valid']['target_node_neg'] 

        neg_valid_edge = torch.stack([source.repeat_interleave(val_neg_edge.size(1)), 
                                val_neg_edge.view(-1)])

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
        test_neg_edge = split_edge['test']['target_node_neg']

        neg_test_edge = torch.stack([source.repeat_interleave(test_neg_edge.size(1)), 
                                test_neg_edge.view(-1)])
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

    # else:
    print('edge size', pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size())
     
    if args.use_heuristic != 'FP':
        pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred,  pos_train_pred = get_prediction(A, \
                full_A, use_heuristic, pos_valid_edge, neg_valid_edge, pos_test_edge, neg_test_edge, pos_train_edge, args)
        get_test_hist(args, full_A, pos_test_pred, neg_test_pred,  use_heuristic)
    else:
        pos_test_pred, neg_test_pred = get_fp_prediction(data.x, pos_test_edge, neg_test_edge, args)
         
        
    state = {
        'pos_val': pos_val_pred,
        'neg_val': neg_val_pred,
        'pos_test': pos_test_pred,
        'neg_test': neg_test_pred,
        'pos_train': pos_train_pred
    }
    
    save_path = args.output_dir + '/beta'+ str(args.beta) + '_pathlen'+ str(args.path_len) + '_' + 'save_score'
    # torch.save(state, save_path)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    if args.data_name == 'ogbl-citation2':
       
        neg_val_pred = neg_val_pred.view(-1, val_neg_edge.size(1))
        neg_test_pred = neg_test_pred.view(-1, test_neg_edge.size(1))
        print('pred size', pos_val_pred.size(), neg_val_pred.size(), pos_test_pred.size(), neg_test_pred.size())

        results = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
        print('heurisitic: ', args.use_heuristic) 
          
        for key, result in results.items():
            train_hits, valid_hits, test_hits = result
            print(key)
            print( f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
        # print('valid/test mrr of ' + args.data_name + ' is: ', result['MRR'][0], result['MRR'][1])
    else:
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, args.use_mrr)

        print('heurisitic: ', args.use_heuristic)    

        if args.use_mrr:
            print('valid/test mrr of ' + args.data_name + ' is: ', result['MRR'][0], result['MRR'][1])

        print('\n')
        
        for i in range(2):
            if i == 0: print('validation performance: ')
            if i == 1: print('test performance: ')

            
            print('hit 20, 50, 100 of ' + args.data_name + ' is: ', result['Hits@20'][i], result['Hits@50'][i], result['Hits@100'][i])


            print('AUC and AP of ' + args.data_name + ' is: ', result['AUC'][i], result['AP'][i] )
            print('\n')

         


if __name__ == "__main__":
   

    main()