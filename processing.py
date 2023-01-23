from utils.processing_utils import *

import numpy as np
from numpy import asarray, save
import math

import xarray as xr

data_path = 'data/'

# Read the datasets.

era5_19592022 = xr.open_dataset(data_path + 'era5_sst_t2m_011959_112022.nc', decode_times=False)
era5_19501978 = xr.open_dataset(data_path + 'era5_sst_t2m_011950_121978.nc', decode_times=False)
print('ERA5:', era5_19592022, era5_19501978)
print('--------------------')
print()

"""
# Turn datasets into arrays and get wanted variables.

era5_19592022 = era5_19592022.to_array()
era5_19501978 = era5_19501978.to_array()

# Convert Kelvin to Celsius.
era5_19592022 -= 273.15
era5_19501978 -= 273.15

print('ERA5:', era5_19592022, era5_19501978)
print('ERA5's shape:', era5_19592022.shape, era5_19501978.shape)
print('--------------------')
print()

era5_sst_19592022 = np.array(era5_19592022[1,:-2,0,:,:]) # Fields at the last two time steps.
era5_sst_19501978 = np.array(era5_19501978[1,:,:,:])
print('ERA5 SST:', era5_sst_19592022, era5_sst_19501978)
print('ERA5 SST's shape:', era5_sst_19592022.shape, era5_sst_19501978.shape)
print('--------------------')
print()

era5_t2m_19592022 = np.array(era5_19592022[0,:-2,0,:,:])
era5_t2m_19501978 = np.array(era5_19501978[0,:,:,:])
print('ERA5 T2M:', era5_t2m_19592022, era5_t2m_19501978)
print('ERA5 T2M's shape:', era5_t2m_19592022.shape, era5_t2m_19501978.shape)
print('--------------------')
print()

era5_sst = np.concatenate((era5_sst_19501978, era5_sst_19592022[240:]))
era5_t2m = np.concatenate((era5_t2m_19501978, era5_t2m_19592022[240:]))

save(data_path + 'era5_sst_grid.npy', era5_sst)
save(data_path + 'era5_t2m_grid.npy', era5_t2m)

print('Save the two tensors in NPY files.')
print('--------------------')
print()

# Calculate SSTAs and T2MAs.

train_split = 0.8
num_year = era5_sst.shape[0] / 12
train_num_year = math.ceil(num_year * train_split)
test_num_year = int(num_year - train_num_year)

print('The number of years for training:', train_num_year)
print('The number of years for testing:', test_num_year)
print('----------')
print()

era5_ssta = []
for row in era5_sst:
  era5_ssta.append(get_ssta(row, train_num_year))
era5_ssta = np.array(era5_ssta)

era5_t2ma = []
for row in era5_t2m:
  era5_t2ma.append(get_ssta(row, train_num_year))
era5_t2ma = np.array(era5_t2ma)

print('ERA5 SSTA:', era5_ssta)
print('ERA5 SSTA's shape:', era5_ssta.shape)
print('ERA5 T2MA:', era5_t2ma)
print('ERA5 T2MA's shape:', era5_t2ma.shape)

save(data_path + 'era5_ssta_grid.npy', era5_ssta)
save(data_path + 'era5_t2ma_grid.npy', era5_t2ma)

print('Save the two tensors in NPY files.')
print('--------------------')
print()

# Combine the two grids into one grid, explained on Page 17, Taylor & Feng, 2022.

era5_combined = era5_sst
mask = np.isnan(era5_sst) & ~np.isnan(era5_t2m)
era5_combined[mask] = era5_t2m[mask]

save(data_path + 'era5_sst_t2m_grid.npy', era5_combined)

print('ERA5 SSTA-T2M:', era5_combined)
print('ERA5 SSTA-T2M's shape:', era5_combined.shape)

print('Save the tensor in an NPY file.')
print('--------------------')
print()
"""

# Else, do it on NC files directly.
# Count the number of non-NA values.
#print(era5_19592022.count())
sst_t2m_19592022 = era5_19592022.fillna(era5_19592022['t2m'])
#print(sst_t2m_19592022)
#print(sst_t2m_19592022.count())
sst_t2m_19501978 = era5_19501978.fillna(era5_19501978['t2m'])

sst_t2m_19592022.to_netcdf(path=data_path + 'era5_sstwt2m_011959_112022.nc')
sst_t2m_19501978.to_netcdf(path=data_path + 'era5_sstwt2m_011950_121978.nc')

print('Save the netCDF datasets in NC files.')
print('--------------------')
print()