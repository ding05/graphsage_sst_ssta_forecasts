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

node_feat_filename = 'node_feats_ssta.npy'
node_y_filename = 'node_feats_ssta.npy'
adj_filename = 'adj_mat_0.3.npy'
#adj_filename = 'adj_mat_0.9_directed.npy'

window_size = 12
lead_time = 1
learning_rate = 0.05 # 0.01 for SSTs # 0.0005, 0.001 for RMSProp for SSTs
weight_decay = 0.0001 # 0.0001 for RMSProp
momentum = 0.9
l1_ratio = 1
num_epochs = 200 #20
# Early stopping, if the validation MSE has not improved for "patience" epochs, stop training.
patience = 20
min_val_mse = np.inf

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

node_y_grid = load(data_path + node_y_filename)

"""
# Convert Kelvin to Celsius.
node_feat_grid -= 273.15
print('Node feature grid in Celsius:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()
"""

# Normalize the data to [-1, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:840])) / (np.max(node_feat_grid[:,:840]) - np.min(node_feat_grid[:,:840])) * 2 - 1
# Normalize the data to [0, 1].
#node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:840])) / (np.max(node_feat_grid[:,:840]) - np.min(node_feat_grid[:,:840]))
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
        #y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
        y.append(node_y_grid[node_i][time_i + window_size + lead_time - 1])
        '''
        # The outputs are normalized node features.
        y.append(node_feat_grid_normalized[node_i][time_i + window_size + lead_time - 1])
        '''
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

# Split the data, following Taylor & Feng, 2022. Use the first 840 time steps.
# Updated 03/2023, the new ERA5 dataset started from 1940. Use the first 840 time steps (1940-2009).

train_graph_list = graph_list[:840]
val_graph_list = graph_list[840:]
test_graph_list = graph_list[840:]

#test_node_feats = node_feat_grid[:, 840 + window_size - lead_time + 1:]
#test_node_feats = node_feat_grid_normalized[:, 840 + window_size - lead_time + 1:]
test_node_feats = node_y_grid[:, 840 + window_size - lead_time + 1:]

# Define the model.
#model, model_class = MultiGraphGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'GCN'
#model, model_class = MultiGraphGAT(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_heads=8, num_graphs=len(train_graph_list)), 'GAT'
model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'SAGE'
#model, model_class = MultiGraphGGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'GGCN'
# If directed graphs
#model, model_class = MultiGraphRGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=50, out_channels=1, num_relations=2, num_bases=4), 'RGCN'

# Define the loss function.
criterion = nn.MSELoss()

# Define the optimizer.
#optimizer = Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, weight_decay=weight_decay, momentum=momentum)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train a multi-graph GCN model, using and modifying the code generated by ChatGPT-3.

print('Start training.')
print('----------')
print()

# Start time
start = time.time()

# Record the results by epoch.
loss_epochs = []
val_mse_nodes_epochs = []
# Early stopping starting counter
counter = 0

for epoch in range(num_epochs):
    # Iterate over the training data.
    for data in train_graph_list:
        optimizer.zero_grad()
        output = model([data])
        loss = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
        """
        # Elastic net
        l1_reg = 0.0
        l2_reg = 0.0
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
        loss += weight_decay * (l1_ratio * l1_reg + (1 - l1_ratio) * l2_reg)
        """
        loss.backward()
        optimizer.step()
    loss_epochs.append(loss.item())

    # Compute the MSE on the validation set.
    with torch.no_grad():
        val_mse_nodes = 0
        pred_node_feat_list = []
        for data in val_graph_list:
            output = model([data])
            val_mse = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
            print('Val predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
            print('Val observations:', [round(i, 4) for i in torch.tensor(data.y).squeeze().tolist()[::300]])
            val_mse_nodes += val_mse
            
            # The model output graph by graph, but we are interested in time series at node by node.
            # Transform the shapes.
            pred_node_feat_list.append(output.squeeze())
        
        val_mse_nodes /= len(val_graph_list)
        val_mse_nodes_epochs.append(val_mse_nodes.item())
        
        pred_node_feat_tensor = torch.stack([tensor for tensor in pred_node_feat_list], dim=1)
        pred_node_feats = pred_node_feat_tensor.numpy()
        gnn_mse = np.mean((pred_node_feats - test_node_feats) ** 2, axis=1)
              
    print('----------')
    print()

    # Print the current epoch and validation MSE.
    print('Epoch [{}/{}], Loss: {:.4f}, Validation MSE (calculated by column / graph): {:.4f}'.format(epoch + 1, num_epochs, loss.item(), val_mse_nodes))
    print('MSEs by node:', gnn_mse)
    print('Validation MSE (calculated by row / time series at nodes): {:.4f}'.format(np.mean(gnn_mse)))
    print('Loss by epoch:', loss_epochs)
    print('Validation MSE by epoch:', val_mse_nodes_epochs)

    # Update the best model weights if the current validation MSE is lower than the previous minimum.
    if val_mse_nodes.item() < min_val_mse:
        min_val_mse = val_mse_nodes.item()
        best_epoch = epoch
        best_model_weights = model.state_dict()
        best_optimizer_state = optimizer.state_dict()
        best_loss = loss
        best_pred_node_feats = pred_node_feats
        counter = 0
    else:
        counter += 1
    # If the validation MSE has not improved for "patience" epochs, stop training.
    if counter >= patience:
        print(f'Early stopping at Epoch {epoch} with best validation MSE: {min_val_mse} at Epoch {best_epoch}.')
        break

print('----------')
print()

# End time
stop = time.time()

print(f'Complete training. Time spent: {stop - start} seconds.')
print('----------')
print()

"""
# Test the model.
with torch.no_grad():
    test_mse_nodes = 0
    for data in test_graph_list:
        output = model([data])
        test_mse = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
        print('Test predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
        print('Test observations:', [round(i, 4) for i in torch.tensor(data.y).squeeze().tolist()[::300]])
        test_mse_nodes += test_mse
    test_mse_nodes /= len(test_graph_list)
    print('Test MSE: {:.4f}'.format(test_mse_nodes))

print('----------')
print()
"""

# Save the results.
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_losses' + '.npy', np.array(loss_epochs))
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valmses' + '.npy', np.array(val_mse_nodes_epochs))
#save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_preds' + '.npy', pred_node_feats)
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_preds' + '.npy', best_pred_node_feats)
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_testobs' + '.npy', test_node_feats)

print('Save the results in NPY files.')
print('----------')
print()

# Save the model.
torch.save({
            #'epoch': num_epochs,
            #'model_state_dict': model.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            'epoch': best_epoch,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict': best_optimizer_state,
            #'loss': loss
            'loss': best_loss
            }, models_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop))

print('Save the checkpoint in a TAR file.')
print('----------')
print()