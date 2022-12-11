from utils.processing_utils import *

import numpy as np
from numpy import asarray, save
import math

import xarray as xr

data_path = "data/"

# Read the dataset.

era5 = xr.open_dataset(data_path + "era5_sst_t2m.nc", decode_times=False)
print("ERA5:")
print(era5)
print("--------------------")
print()

# Turn it into a smaller size.

era5 = era5.to_array()
#print("ERA5:", era5)
print("ERA5's shape:", era5.shape)

era5_sst = np.array(era5[1,:,1,:,:])
#print("ERA5 SST:", era5_sst)
print("ERA5 SST's shape:", era5_sst.shape)

era5_t2m = np.array(era5[0,:,1,:,:])
#print("ERA5 T2M:", era5_t2m)
print("ERA5 T2M's shape:", era5_t2m.shape)

save(data_path + "era5_sst_grid.npy", era5_sst)
save(data_path + "era5_t2m_grid.npy", era5_t2m)

print("Save the two tensors in NPY files.")
print("--------------------")
print()

# Calculate SSTAs and T2MAs.

train_split = 0.8
num_year = era5_sst.shape[0] / 12
train_num_year = math.ceil(num_year * train_split)
test_num_year = int(num_year - train_num_year)

print("The number of years for training:", train_num_year)
print("The number of years for testing:", test_num_year)
print("----------")
print()

era5_ssta = []
for row in era5_sst:
  era5_ssta.append(get_ssta(row, train_num_year))
era5_ssta = np.array(era5_ssta)

era5_t2ma = []
for row in era5_t2m:
  era5_t2ma.append(get_ssta(row, train_num_year))
era5_t2ma = np.array(era5_t2ma)

print("ERA5 SSTA:", era5_ssta)
print("ERA5 SSTA's shape:", era5_ssta.shape)
print("ERA5 T2MA:", era5_t2ma)
print("ERA5 T2MA's shape:", era5_t2ma.shape)

save(data_path + "era5_ssta_grid.npy", era5_ssta)
save(data_path + "era5_t2ma_grid.npy", era5_t2ma)

print("Save the two tensors in NPY files.")
print("--------------------")
print()