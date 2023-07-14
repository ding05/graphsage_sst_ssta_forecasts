from utils.process_utils import *

import numpy as np
from numpy import asarray, save, load
import math

import xarray as xr

data_path = 'data/'

train_num_year = 840

"""
# Read the dataset from NetCDF files.

era5_19402022_globe = xr.open_dataset(data_path + 'era5_sst_011940_122022_globe.nc', decode_times=False)
print('ERA5 Globe:', era5_19402022_globe)
print('--------------------')
print()

# Convert SSTs into SSTAs and save them into NetCDF files.

sst_19402022 = era5_19402022_globe['sst']
sst_19402022_np = sst_19402022.values

print(sst_19402022_np)
print("ERA5 Globe's shape:", sst_19402022_np.shape)
print('--------------------')
print()

ssta_19402022_np = np.empty(sst_19402022_np.shape)

for lat_idx in range(sst_19402022_np.shape[1]):
    for lon_idx in range(sst_19402022_np.shape[2]):
        time_series = sst_19402022_np[:, lat_idx, lon_idx]
        
        if np.all(np.isnan(time_series)):
            ssta_19402022_np[:, lat_idx, lon_idx] = np.nan
            continue
        
        new_time_series = get_ssta(time_series, train_num_year)
        ssta_19402022_np[:, lat_idx, lon_idx] = new_time_series

print(ssta_19402022_np)
print("ERA5 SSTA Globe's shape:", ssta_19402022_np.shape)
print('--------------------')
print()

# Write the new dataset.

era5_19402022_ssta_globe = era5_19402022_globe
era5_19402022_ssta_globe['sst'].values = ssta_19402022_np

era5_19402022_ssta_globe.to_netcdf(data_path + 'era5_ssta_011940_122022_globe.nc')
"""

# Read the dataset from NumPy files.

data_path = 'data/'
node_filename = 'node_feats_sst.npy'

node_feat_grid = load(data_path + node_filename)
print('ERA5 Globe SSTs in NumPy:', node_feat_grid)
print('----------')
print("ERA5 Globe SSTs' shape:", node_feat_grid.shape)
print('--------------------')

# Convert SSTs into SSTAs and save them into NumPy files.

"""
converted_node_filename = 'node_feats_ssta.npy'

converted_node_feat_grid = []
for row in node_feat_grid:
  converted_node_feat_grid.append(get_ssta(row, train_num_year))
converted_node_feat_grid = np.array(converted_node_feat_grid)

print('ERA5 Globe SSTAs in NumPy:', converted_node_feat_grid)
print('----------')
print("ERA5 Globe SSTAs' shape:", converted_node_feat_grid.shape)
print('--------------------')

np.save(data_path + converted_node_filename, converted_node_feat_grid)
"""

converted_node_filename = 'node_feats_ssta_1980_2010.npy'

converted_node_feat_grid = []
for row in node_feat_grid:
  converted_node_feat_grid.append(get_ssta_1980_2010(row))
converted_node_feat_grid = np.array(converted_node_feat_grid)

print('ERA5 Globe SSTAs in NumPy:', converted_node_feat_grid)
print('----------')
print("ERA5 Globe SSTAs' shape:", converted_node_feat_grid.shape)
print('--------------------')

np.save(data_path + converted_node_filename, converted_node_feat_grid)