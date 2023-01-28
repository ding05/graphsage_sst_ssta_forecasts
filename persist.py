import numpy as np
from numpy import asarray, save, load

data_path = 'data/'
node_filename = 'node_feats.npy'

# Load the data.

node_feats_grid = load(data_path + node_filename)
print('Node feature grid in Kelvin:', node_feats_grid)
print('Shape:', node_feats_grid.shape)
print('----------')
print()

# Convert Kelvin to Celsius.
node_feats_grid -= 273.15
print('Node feature grid in Celsius:', node_feats_grid)
print('Shape:', node_feats_grid.shape)
print('----------')
print()

persist_pred = node_feats_grid[:, 760:-1]
persist_obs = node_feats_grid[:, 761:]

persist_mse = np.mean((persist_pred - persist_obs) ** 2, axis=1)

print('MSEs by node:', persist_mse)
print('Average MSE:', np.mean(persist_mse))