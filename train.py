from utils.unet_lstm import *

import numpy as np
from numpy import asarray, save, load
import math

import xarray as xr

data_path = "data/"
models_path = "configs/"
out_path = "out/"

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

# Set up the Unet-LSTM.

train_split = 0.9
num_year = grid_norm.shape[0] / 12
train_num_year = math.ceil(num_year * train_split)
test_num_year = int(num_year - train_num_year)

time_sequence = train_num_year * 12
print("Number of training time steps:", time_sequence)

num_lats = grid_norm.shape[1]
num_longs = grid_norm.shape[2]
bound = 0

num_hidden_units = 16

num_features = 12
num_responses = 2

# Train the model.

# Initialize Horovod.
hvd.init()

unet_lstm(num_hidden_units, num_responses, time_sequence, num_features, num_lats, num_longs, bound)