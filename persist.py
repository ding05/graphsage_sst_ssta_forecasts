import numpy as np
from numpy import asarray, save, load

data_path = 'data/'
out_path = 'out/'

node_filename = 'node_feats_ssta.npy'

window_size = 12
lead_time = 1

# Load the data.

node_feat_grid = load(data_path + node_filename)
print('Node feature grid in Kelvin:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()

"""
# Convert Kelvin to Celsius.
node_feat_grid -= 273.15
print('Node feature grid in Celsius:', node_feat_grid)
print('Shape:', node_feat_grid.shape)
print('----------')
print()
"""

# Normalize the data to [-1, 1].
node_feat_grid_normalized = (node_feat_grid - np.min(node_feat_grid)) / (np.max(node_feat_grid) - np.min(node_feat_grid)) * 2 - 1
print('Normalized node feature grid:', node_feat_grid_normalized)
print('Shape:', node_feat_grid_normalized.shape)
print('----------')
print()

#persist_preds = node_feat_grid[:, 840 + window_size - lead_time :-1]
persist_preds = node_feat_grid_normalized[:, 840 + window_size - lead_time :-1]
#persist_obs = node_feat_grid[:, 840 + window_size - lead_time + 1:]
persist_obs = node_feat_grid_normalized[:, 840 + window_size - lead_time + 1:]

persist_mses = np.mean((persist_preds - persist_obs) ** 2, axis=1)

print('MSEs by node:', persist_mses)
print('Average MSE:', np.mean(persist_mses))
print('----------')
print()

# Save the results.

save(out_path + 'persist_' + 'preds' + '.npy', persist_preds)
save(out_path + 'persist_' + 'obs' + '.npy', persist_obs)
save(out_path + 'persist_' + 'mses' + '.npy', persist_mses)

print('Save the results in NPY files.')
print('----------')
print()