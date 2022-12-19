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

# Sliding Window

train_split = 0.9
num_year = grid_norm.shape[0] / 12
train_num_year = math.ceil(num_year * train_split)
test_num_year = int(num_year - train_num_year)

time_sequence = train_num_year * 12
print("Number of training time steps:", time_sequence)

train_grid = grid_norm[:time_sequence]
test_grid = grid_norm[time_sequence:]

window_size = 12
lead_time = 2

train_grid_input = []
train_grid_output = []

for i in range(len(train_grid)-window_size-lead_time+1):
    train_grid_input.append(train_grid[i:i+window_size])
    train_grid_output.append(train_grid[i+window_size+lead_time-2:i+window_size+lead_time])
train_grid_input = np.array(train_grid_input)
train_grid_output = np.array(train_grid_output)
train_grid_input = np.einsum('klij->kijl', train_grid_input)
train_grid_output = np.einsum('klij->kijl', train_grid_output)
train_grid_input = np.expand_dims(train_grid_input, axis=0)
train_grid_output = np.expand_dims(train_grid_output, axis=0)

test_grid_input = []
test_grid_output = []

for i in range(len(test_grid)-window_size-lead_time+1):
    test_grid_input.append(test_grid[i:i+window_size])
    test_grid_output.append(test_grid[i+window_size+lead_time-2:i+window_size+lead_time])
test_grid_input = np.array(test_grid_input)
test_grid_output = np.array(test_grid_output)
test_grid_input = np.einsum('klij->kijl', test_grid_input)
test_grid_output = np.einsum('klij->kijl', test_grid_output)
test_grid_input = np.expand_dims(test_grid_input, axis=0)
test_grid_output = np.expand_dims(test_grid_output, axis=0)

print(test_grid_input.shape)
print(test_grid_output.shape)

# Set up the Unet-LSTM.

time_sequence = train_grid_input.shape[1]

num_lats = grid_norm.shape[1]
num_longs = grid_norm.shape[2]
bound = 0

num_hidden_units = 16

num_features = 12
num_responses = 2

# Train the model.

# Initialize Horovod.
hvd.init()

model = unet_lstm(num_hidden_units, num_responses, time_sequence, num_features, num_lats, num_longs, bound)

history = model.fit(train_grid_input, train_grid_output, batch_size=4, epochs=200, verbose=True)
loss, mse  = model.evaluate(test_grid_input, test_grid_output, verbose=False)