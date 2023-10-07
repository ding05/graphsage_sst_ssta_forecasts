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

saved_model = 'SAGE_0.7_1688867785.0908325'

window_size = 12

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

# Normalize the data to [0, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:852])) / (np.max(node_feat_grid[:,:852]) - np.min(node_feat_grid[:,:852]))
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
num_time = node_feat_grid.shape[1] - window_size - 1 + 1

#for lead_time in [2, 3, 6, 12, 18, 24]:
#for lead_time in [24]:
for lead_time in list(range(1, 25)):
    
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
            y.append(node_feat_grid_normalized[node_i][time_i + window_size + 1 - 1])
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
    test_graph_list = graph_list[840:]
    
    test_node_feats = node_feat_grid_normalized[:, 840 + window_size + lead_time - 1:]
    print("Observations' shape:", test_node_feats.shape)
    print('----------')
    print()
    
    # Define the model.
    model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE'

    # Load the model.
    checkpoint = torch.load(models_path + saved_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Pre-trained model loaded")
    print('----------')
    print()
    
    # Initialization.
    predictions = []
    
    # Use a loop to iterate over every starting time.
    for time in range(node_feat_grid.shape[1] - (840 + window_size + lead_time - 1)):
    
        # Extract strating test input features.
        start_test_input_graph_list = [graph_list[840 + time]]
        
        # Initialization.
        test_input_graph_list = start_test_input_graph_list
    
        # Use a loop to input the features and update the features.
        for month in range(lead_time):
        
            print('Month', month)
            print('----------')
            print()
            
            output = model(test_input_graph_list)
            #print('Predictions of the first three nodes:', [round(i, 4) for i in output.squeeze().tolist()[:3]])
            print('Predictions of every 300th node:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
            print("Predictions' shape:", output.squeeze().shape)
            print('----------')
            print()
            
            # Update the graph with new predictions.
            """
            print('x[0]:', test_input_graph_list[0].x[0])
            print("x'shape:", test_input_graph_list[0].x.shape)
            """
            print('x:', test_input_graph_list[0].x)
            print('----------')
            print()
            
            old_x = test_input_graph_list[0].x
            added_x = output
            new_x = torch.cat((old_x, added_x), dim=1)
            new_x = new_x[:, 1:]
            new_graph = Data(x=new_x, y=None, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
            test_input_graph_list = [new_graph]
            print('new x:', test_input_graph_list[0].x)
            print('----------')
            print()
            
        # Save the predictions.
        predictions.append(output)
        print("Predictions' length:", len(predictions))
        print('--------------------')
        print()
    
    # Save the results.
    prediction_tensor = torch.stack(predictions, dim=1)
    prediction_array = prediction_tensor.detach().numpy()
    prediction_array = prediction_array.squeeze()
    print("Predictions' shape:", prediction_array.shape)
    print("Observations' shape:", test_node_feats.shape)
    print('Average MSE:', ((prediction_array - test_node_feats) ** 2).mean())
    print('--------------------')
    print()
    
    save(out_path + saved_model + '_' + str(lead_time) + '_preds.npy', prediction_array)
    save(out_path + saved_model + '_' + str(lead_time) + '_testobs.npy', test_node_feats)