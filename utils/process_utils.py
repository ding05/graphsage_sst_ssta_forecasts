import numpy as np
import pandas as pd
import torch

# Drop the land nodes (the rows in the node feature matrix with NAs).
def drop_rows_w_nas(arr, *args, **kwarg):
    assert isinstance(arr, np.ndarray)
    dropped=pd.DataFrame(arr).dropna(*args, **kwarg).values
    if arr.ndim==1:
        dropped=dropped.flatten()
    return dropped

# Get SSTAs from an SST vector.
def get_ssta(time_series, train_num_year):
    monthly_avg = []
    for month in range(12):
      monthly_sst = time_series[month:train_num_year*12:12]
      monthly_avg.append(avg(monthly_sst))
      time_series[month::12] -= monthly_avg[month]
    return time_series

# Get SSTAs from an SST vector using the means of certain years.
def get_ssta_1980_2010(time_series):
    monthly_avg = []
    for month in range(12):
      monthly_sst = time_series[480+month:852:12]
      monthly_avg.append(avg(monthly_sst))
      time_series[month::12] -= monthly_avg[month]
    return time_series

# Extract output vectors for more places.
def extract_y(lat, lon, filename, data_path):
    soda_temp = soda.loc[dict(LAT=str(lat), LONN359_360=str(lon))]
    soda_temp_sst = np.zeros((len(soda.TIME), 1))
    soda_temp_sst[:,:] = soda_temp.variables['TEMP'][:,:]
    soda_temp_ssta = get_ssta(soda_temp_sst)
    save(data_path + 'y_' + filename + '.npy', soda_temp_ssta)

def avg(list):
    return sum(list) / len(list)

# Sort the edge index tensor to be column-wise for the LSTM aggregator. 
def sort_by_destination(edge_index):
    edges = edge_index.t()
    sorted_indices = torch.argsort(edges[:, 1], dim=0)
    sorted_edges = edges[sorted_indices]
    sorted_edge_index = sorted_edges.t()
    return sorted_edge_index

# Calculate the critical success index (CSI) node by node.
def calculate_csi(pred_feats, test_feats, threshold):
    pred_pos = pred_feats > threshold
    true_pos = test_feats > threshold
    TP = np.logical_and(pred_pos, true_pos).sum()
    FP = np.logical_and(pred_pos, ~true_pos).sum()
    FN = np.logical_and(~pred_pos, true_pos).sum()
    # Avoid division by zero.
    denominator = TP + FP + FN
    if denominator == 0:
        return np.nan
    else:
        return TP / denominator