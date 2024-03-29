from utils.gnns import *
from utils.loss_funcs import *
from utils.process_utils import *
from utils.eval_utils import *

import numpy as np
from numpy import asarray, save, load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index

import time

data_path = 'data/'
models_path = 'configs/'
out_path = 'out/'

#node_feat_filename = 'node_feats_sst.npy'
#node_feat_filename = 'node_feats_ssta.npy'
node_feat_filename = 'node_feats_ssta_1980_2010.npy'
#node_y_filename = 'node_feats_ssta.npy'
#node_y_filename = 'blob.npy'
#graph_y_filename = 'blob.npy'
adj_filename = 'adj_mat_0.9.npy'
#adj_filename = 'adj_mat_0.9_100.npy'
#adj_filename = 'adj_mat_0.9_directed.npy'

window_size = 12
lead_time = 1
learning_rate = 0.001 # 0.001 for SSTs with MSE # 0.0005, 0.001 for RMSProp for SSTs
#learning_rate = 0.01 # For the GraphSAGE-LSTM
#learning_rate = 0.01 # For the BMSE loss with the noise variable greater than 1.0
weight_decay = 0.0001 # 0.0001 for RMSProp
momentum = 0.9
l1_ratio = 1
num_epochs = 400 #1000, 400, 200
# Early stopping, if the validation MSE has not improved for "patience" epochs, stop training.
patience = num_epochs #100, 40, 20
min_val_mse = np.inf
# For the GraphSAGE-LSTM
sequence_length = 12

# Load the data.

node_feat_grid = load(data_path + node_feat_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

"""
node_y_grid = load(data_path + node_y_filename)
y_seq = load(data_path + graph_y_filename)
node_y_grid = y_seq.reshape(-1, 1)
node_y_grid = np.tile(node_y_grid, (1, node_feat_grid.shape[0]))
node_y_grid = node_y_grid.T
print('Graph label grid:', node_y_grid)
print('Shape:', node_y_grid.shape)
print('----------')
print()
"""

"""
# Convert Kelvin to Celsius.
node_feat_grid -= 273.15
print('Node feature grid in Celsius:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()
"""

# Normalize the data to [-1, 1].
#node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid[:,:852])) / (np.max(node_feat_grid[:,:852]) - np.min(node_feat_grid[:,:852])) * 2 - 1
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
        # The outputs are node labels.
    x = torch.tensor(x)
    # Generate incomplete graphs with the adjacency matrix.
    edge_index = torch.tensor(adj_mat, dtype=torch.long)
    # Sort the edge index for the LSTM aggregator for the GraphSAGE.
    #edge_index = sort_by_destination(edge_index)
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # Generate complete graphs.
    #node_indices = torch.arange(node_feat_grid.shape[0])
    #combinations = torch.combinations(node_indices, r=2) 
    #edge_index = combinations.t().contiguous()
    #data = Data(x=x, y=y, edge_index=edge_index, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # If an empty adjacency matrix
    #data = Data(x=x, y=y, num_nodes=node_feat_grid.shape[0], num_edges=adj_mat.shape[1], has_isolated_nodes=True, has_self_loops=False, is_undirected=True)
    # If directed graphs (Comment "edge_index, _ = sort_edge_index(edge_index)".)
    #edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)
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

# Split the data.
# Updated 03/2023, the new ERA5 dataset started from 1940. Use the first 840 time steps (1941-2010) of the graph set for training.

train_graph_list = graph_list[:840]
val_graph_list = graph_list[840:]
test_graph_list = graph_list[840:]

#test_node_feats = node_feat_grid[:, 840 + window_size - lead_time + 1:]
test_node_feats = node_feat_grid_normalized[:, 840 + window_size + lead_time - 1:]
#test_node_feats = node_y_grid[:, 840 + window_size - lead_time + 1:]

# Compute the percentiles using the training set only.
node_feats_90 = np.percentile(node_feat_grid[:, :840], 90, axis=1)
node_feats_95 = np.percentile(node_feat_grid[:, :840], 95, axis=1)
node_feats_normalized_90 = np.percentile(node_feat_grid_normalized[:, :840], 90, axis=1)
node_feats_normalized_95 = np.percentile(node_feat_grid_normalized[:, :840], 95, axis=1)

# Select one threshold array.
threshold_tensor = torch.tensor(node_feats_normalized_90).float()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#threshold_tensor = torch.tensor(node_feats_normalized_90).float().to(device)

# Define the model.
#model, model_class = MultiGraphGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'GCN'
#model, model_class = MultiGraphGAT(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_heads=8, num_graphs=len(train_graph_list)), 'GAT'
model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE'
#model, model_class = MultiGraphSage_LSTM(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE_LSTM'
#model, model_class = MultiGraphSage(in_channels=graph_list[0].x[0].shape[0], hid_channels=15, out_channels=1, num_graphs=len(train_graph_list), aggr='mean'), 'SAGE_Blob'
#model, model_class = MultiGraphGGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=30, out_channels=1, num_graphs=len(train_graph_list)), 'GGCN'
# If directed graphs
#model, model_class = MultiGraphRGCN(in_channels=graph_list[0].x[0].shape[0], hid_channels=50, out_channels=1, num_relations=2, num_bases=4), 'RGCN'

# Define the loss function.
criterion = BMCLoss(0.002)
criterion_test = nn.MSELoss()

# Define the optimizer.
#optimizer = Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9, weight_decay=weight_decay, momentum=momentum)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train a multi-graph GNN model.

print('Start training.')
print('----------')
print()

# Start time
start = time.time()

# Record the results by epoch.
loss_epochs = []
val_mse_nodes_epochs = []
val_precision_nodes_epochs = []
val_recall_nodes_epochs = []
val_csi_nodes_epochs = []
noise_var_epochs = []
# Early stopping starting counter
counter = 0

for epoch in range(num_epochs):
    # Iterate over the training data.
    if model_class != 'SAGE_LSTM':
        for data in train_graph_list:
            optimizer.zero_grad()
            output = model([data])
            loss, noise_var = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
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
    else:
        if epoch == 0:
            print('Special case: GraphSAGE-LSTM')
            print('----------')
            print()
        for i in range(0, len(train_graph_list) - sequence_length):
            data_sequence = train_graph_list[i:i+sequence_length]
            target_data = train_graph_list[i+sequence_length]
            optimizer.zero_grad()
            output = model([data_sequence])
            loss, noise_var = criterion(output.squeeze(), torch.tensor(data.y).squeeze())
            loss.backward()
            optimizer.step()
    loss_epochs.append(loss.item())
    noise_var_epochs.append(noise_var.item())

    # Compute the MSE, precision, recall, and critical success index (CSI) on the validation set.
    with torch.no_grad():
        val_mse_nodes = 0
        val_precision_nodes = 0
        val_recall_nodes = 0
        val_csi_nodes = 0
        pred_node_feat_list = []
        
        if model_class != 'SAGE_LSTM':
            for data in val_graph_list:
                output = model([data])
                val_mse = criterion_test(output.squeeze(), torch.tensor(data.y).squeeze())
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
            
            # Precision
            val_precision_nodes = np.nanmean([calculate_precision(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_precision_nodes_epochs.append(val_precision_nodes.item())
            # Recall
            val_recall_nodes = np.nanmean([calculate_recall(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_recall_nodes_epochs.append(val_recall_nodes.item())
            # CSI
            val_csi_nodes = np.nanmean([calculate_csi(pred_node_feats[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_csi_nodes_epochs.append(val_csi_nodes.item())
        
        else:
            for i in range(0, len(val_graph_list) - sequence_length):
                data_sequence = val_graph_list[i:i+sequence_length]
                target_data = val_graph_list[i+sequence_length]
                output = model([data_sequence])
                val_mse = criterion_test(output, torch.tensor(target_data.y).squeeze())
                print('Val predictions:', [round(i, 4) for i in output.squeeze().tolist()[::300]])
                print('Val observations:', [round(i, 4) for i in torch.tensor(target_data.y).squeeze().tolist()[::300]])
                val_mse_nodes += val_mse
                
                pred_node_feat_list.append(output.squeeze())

            val_mse_nodes /= len(val_graph_list)
            val_mse_nodes_epochs.append(val_mse_nodes.item())
            
            pred_node_feat_tensor = torch.stack([tensor for tensor in pred_node_feat_list], dim=1)
            pred_node_feats = pred_node_feat_tensor.numpy()
            # Introduce aritificial data points to match the length of the test set.
            padding = np.zeros((pred_node_feats.shape[0], sequence_length))
            pred_node_feats_padded = np.concatenate([padding, pred_node_feats], axis=1)
            gnn_mse = np.mean((pred_node_feats_padded - test_node_feats) ** 2, axis=1)

            # Precision
            val_precision_nodes = np.nanmean([calculate_precision(pred_node_feats_padded[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_precision_nodes_epochs.append(val_precision_nodes.item())
            # Recall
            val_recall_nodes = np.nanmean([calculate_recall(pred_node_feats_padded[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_recall_nodes_epochs.append(val_recall_nodes.item())
            # CSI            
            val_csi_nodes = np.nanmean([calculate_csi(pred_node_feats_padded[i], test_node_feats[i], node_feats_normalized_90[i]) for i in range(node_feats_normalized_90.shape[0])])
            val_csi_nodes_epochs.append(val_csi_nodes.item())

    print('----------')
    print()

    # Print the current epoch and validation MSE.
    print('Epoch [{}/{}], Loss: {:.6f}, Validation MSE (calculated by column / graph): {:.6f}'.format(epoch + 1, num_epochs, loss.item(), val_mse_nodes))
    print('MSEs by node:', gnn_mse)
    print('Validation MSE, precision, recall, and CSI (calculated by row / time series at nodes): {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(np.mean(gnn_mse), val_precision_nodes, val_recall_nodes, val_csi_nodes))
    #print('Validation precision (calculated by row / time series at nodes): {:.4f}'.format(val_precision_nodes))
    #print('Validation recall (calculated by row / time series at nodes): {:.4f}'.format(val_recall_nodes))
    #print('Validation CSI (calculated by row / time series at nodes): {:.4f}'.format(val_csi_nodes))
    #print('Loss by epoch:', loss_epochs)
    print('Loss by epoch:', [float('{:.6f}'.format(loss)) for loss in (loss_epochs[-20:] if len(loss_epochs) > 20 else loss_epochs)]) # Print the last 20 elements if the list is too long.
    #print('Noise variable by epoch:', [float('{:.4f}'.format(noise_var)) for noise_var in noise_var_epochs[-20:]])
    print('Noise variable:', float('{:.4f}'.format(noise_var_epochs[-1])))
    print('Validation MSE by epoch:', [float('{:.6f}'.format(val_mse)) for val_mse in (val_mse_nodes_epochs[-20:] if len(val_mse_nodes_epochs) > 20 else val_mse_nodes_epochs)]) # Same as above.
    print('Validation precision by epoch:', [float('{:.6f}'.format(val_precision)) for val_precision in (val_precision_nodes_epochs[-20:] if len(val_precision_nodes_epochs) > 20 else val_precision_nodes_epochs)])
    print('Validation recall by epoch:', [float('{:.6f}'.format(val_recall)) for val_recall in (val_recall_nodes_epochs[-20:] if len(val_recall_nodes_epochs) > 20 else val_recall_nodes_epochs)])
    print('Validation CSI by epoch:', [float('{:.6f}'.format(val_csi)) for val_csi in (val_csi_nodes_epochs[-20:] if len(val_csi_nodes_epochs) > 20 else val_csi_nodes_epochs)])
    print('Persistence MSE:', ((test_node_feats[:,1:] - test_node_feats[:,:-1])**2).mean())

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
        test_mse = criterion_test(output.squeeze(), torch.tensor(data.y).squeeze())
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
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valprecisions' + '.npy', np.array(val_precision_nodes_epochs))
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valrecalls' + '.npy', np.array(val_recall_nodes_epochs))
save(out_path + model_class + '_' + adj_filename[8:-4] + '_' + str(stop) +  '_valcsis' + '.npy', np.array(val_csi_nodes_epochs))
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