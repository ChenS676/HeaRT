from ogb.linkproppred import PygLinkPropPredDataset
import networkx as nx
import os 
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
import torch_geometric
from matplotlib import pyplot as plt
import torch  
from deepsnap.graph import Graph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset


def visualize(G, color_map=None, seed=123):
  if color_map is None:
    color_map = '#c92506'
  pos=nx.spring_layout(G, seed=seed)
  plt.figure(figsize=(8, 8))
  nodes = nx.draw_networkx_nodes(G, pos=pos, \
                                 label=None, node_color=color_map, node_shape='o', node_size=150)
  edges = nx.draw_networkx_edges(G, pos=pos, alpha=0.5)
  #   if color_map is not None:
  #     plt.scatter([],[], c='#c92506', label='Nodes with label 0', edgecolors="black", s=140)
  #     plt.scatter([],[], c='#fcec00', label='Nodes with label 1', edgecolors="black", s=140)
  #     plt.legend(prop={'size': 13}, handletextpad=0)
  nodes.set_edgecolor('black')
  plt.show()
  
  
if 'IS_GRADESCOPE_ENV' not in os.environ:
#   num_nodes = 100
#   p = 0.05
#   seed = 100

#   # Generate a networkx random graph
#   G = nx.gnp_random_graph(num_nodes, p, seed=seed)

#   # Generate some random node features and labels
#   node_feature = {node : torch.rand([5, ]) for node in G.nodes()}
#   node_label = {node : torch.randint(0, 2, ()) for node in G.nodes()}

#   # Set the random features and labels to G
#   nx.set_node_attributes(G, node_feature, name='node_feature')
#   nx.set_node_attributes(G, node_label, name='node_label')

#   # Print one node example
#   for node in G.nodes(data=True):
#     print(node)
#     break
    data_name = "ogbl-collab"
    if data_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='./dataset', name=data_name)
    else:
      dataset = PygLinkPropPredDataset(name=data_name, root="./dataset")

    data = dataset[0]
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)

    # color_map = ['#c92506' if node[1]['node_label'].item() == 0 else '#fcec00' for node in g.nodes(data=True)]

    # Transform the networkx graph into the deepsnap graph
    graph = Graph(g)

    # Print out the general deepsnap graph information
    print(graph)

    # DeepSNAP will also generate the edge_index tensor
    print("Edge index (edge_index) has shape {} and type {}".format(graph.edge_index.shape, graph.edge_index.dtype))

    # Different from only storing tensors, deepsnap graph also references to the networkx graph
    # We will discuss why the reference will be helpful later
    print("The DeepSNAP graph has {} as the internal manupulation graph".format(type(graph.G)))

# if color_map is None:
#     color_map = '#c92506'

plt.figure(figsize=(18, 18))
pos =  nx.kamada_kawai_layout(g)

nodes = nx.draw_networkx_nodes(g, pos=pos, \
                                label=None, node_color='#c92506', node_shape='o', node_size=150)
edges = nx.draw_networkx_edges(g, pos=pos, alpha=0.4)
#   if color_map is not None:
#     plt.scatter([],[], c='#c92506', label='Nodes with label 0', edgecolors="black", s=140)
#     plt.scatter([],[], c='#fcec00', label='Nodes with label 1', edgecolors="black", s=140)
#     plt.legend(prop={'size': 13}, handletextpad=0)
nodes.set_edgecolor('black')
plt.savefig(f'/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/{data_name}_spring.png')
  

# TODO visualize the graph with different size of vertex of degree

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def visual_degree(G):
  degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
  dmax = max(degree_sequence)

  fig = plt.figure("Degree of a random graph", figsize=(8, 8))
  # Create a gridspec for adding subplots of different sizes
  axgrid = fig.add_gridspec(5, 4)

  ax0 = fig.add_subplot(axgrid[0:3, :])
  Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
  pos = nx.spring_layout(Gcc, seed=10396953)
  nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
  nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
  ax0.set_title("Connected components of G")
  ax0.set_axis_off()

  ax1 = fig.add_subplot(axgrid[3:, :2])
  ax1.plot(degree_sequence, "b-", marker="o")
  ax1.set_title("Degree Rank Plot")
  ax1.set_ylabel("Degree")
  ax1.set_xlabel("Rank")

  ax2 = fig.add_subplot(axgrid[3:, 2:])
  ax2.bar(*np.unique(degree_sequence, return_counts=True))
  ax2.set_title("Degree histogram")
  ax2.set_xlabel("Degree")
  ax2.set_ylabel("# of Nodes")

  fig.tight_layout()
  plt.savefig(f'/pfs/work7/workspace/scratch/cc7738-nlp_graph/HeaRT_Mao/{data_name}_deg.png')
  
visual_degree(g)