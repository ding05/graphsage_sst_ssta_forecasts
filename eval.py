from utils.gnns import *

import numpy as np
from numpy import asarray, save, load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

import time

data_path = 'data/'
models_path = 'configs/'
out_path = 'out/'

node_feat_filename = 'node_feats_sst.npy'
adj_filename = 'adj_mat_0.7.npy'

saved_model = 'SAGE_0.7_1684017063.223997'

window_size = 12
lead_time = 23

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

# Normalize the data to [0, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:840])) / (np.max(node_feat_grid[:,:840]) - np.min(node_feat_grid[:,:840]))
print('Normalized node feature grid:', node_feat_grid_normalized)
print('Shape:', node_feat_grid_normalized.shape)
print('----------')
print()

adj_mat = load(data_path + adj_filename)
print('Adjacency matrix:', adj_mat)
print('Shape:', adj_mat.shape)
print('----------')
print()

# Compute the total number of time steps.
num_time = node_feat_grid.shape[1] - window_size - lead_time + 1

# Generate PyG graphs from NumPy arrays.
graph_list = []
for time_i in range(num_time):
    x = []
    y = []
    for node_i in range(node_feat_grid.shape[0]):
        # The inputs are normalized node features.
        #x.append(node_feat_grid[node_i][time_i : time_i + window_size])
        x.append(node_feat_grid_normalized[node_i][time_i : time_i + window_size])
        # The outputs are node features in Kelvin.
        #y.append(node_feat_grid[node_i][time_i + window_size + lead_time - 1])
        #y.append(node_y_grid[node_i][time_i + window_size + lead_time - 1])
        # The outputs are normalized node features.
        y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
    x = torch.tensor(x)
    # Generate incomplete graphs with the adjacency matrix.
    edge_index = torch.tensor(adj_mat, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # Generate complete graphs.
    #node_indices = torch.arange(node_feat_grid.shape[0])
    #combinations = torch.combinations(node_indices, r=2) 
    #edge_index = combinations.t().contiguous()
    #data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # If an empty adjacency matrix
    #data = Data(x=x, y=y, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # If directed graphs
    edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)
    #data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=False)
    graph_list.append(data)

# Set the number of decimals in torch tensors printed.
torch.set_printoptions(precision=8)

print('Inputs of the first node in the first graph, i.e. the first time step:', graph_list[0].x[0])
#print('Output of the first node in the first graph:', graph_list[0].y[0])
print('Check if they match those in the node features:', node_feat_grid[0][:13])
#print('Check if they match those in the node features:', node_feat_grid[0][:13])
print('Check if they match those in the normalized node features:', node_feat_grid_normalized[0][:13])
print('----------')
print()

train_graph_list = graph_list[:840]
test_graph_list = graph_list[840 : 840 + lead_time]
#print('Test output observations:', test_graph_list)
print("Test output observations' length:", len(test_graph_list)) # The list contains lead_time graphs.
print('----------')
print()

# Extract strating test input features.
test_input_graph_list = [graph_list[840]]
#print('Starting test input features:', test_input_graph_list)
print("Starting test input features' length:", len(test_input_graph_list)) # The list contains window_size graphs.
print('----------')
print()

# Define the model.
model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'SAGE'

# Load the model.
checkpoint = torch.load(models_path + saved_model)
model.load_state_dict(checkpoint['model_state_dict'])
print("Pre-trained model loaded")
print('----------')
print()

# Use a loop to input the features and update the features.
#for month in range(lead_time):
for month in range(1):
    for data in test_input_graph_list:
        output = model([data])
        print('Predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
        print("Predictions' shape:", output.squeeze().shape)