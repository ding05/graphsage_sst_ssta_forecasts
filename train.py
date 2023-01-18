from utils.unet_lstm import *
from keras_dgl.layers import GraphCNN, GraphConvLSTM

import numpy as np
from numpy import asarray, save, load
import math

import xarray as xr

import time

from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# Load the data.

era5_19592022 = xr.open_dataset(data_path + "era5_sstwt2m_011950_082022_globe.nc", decode_times=False)
print(era5_19592022)

era5_19592022 = era5_19592022.to_array()

# Convert Kelvin to Celsius.
#era5_19592022 -= 273.15

grid = np.array(era5_19592022[0,:,:,:])

print(grid[-1])
print(grid.shape)

# Normalize the data.

grid_norm = grid / np.linalg.norm (grid)

print(grid_norm[-1])
print(grid_norm.shape)

for lead_time in range(1, 25):
    print()