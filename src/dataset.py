import torch
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, random_split, DataLoader

def get_dataloaders(seed, batch_size):
  # =============== Build data loaders ======================================#
  tqdm.write("Building data loaders...")

  path_to_h5_train, path_to_csv_train, path_to_records = 'codesubset/train.h5', 'codesubset/train.csv', 'codesubset/train/RECORDS.txt'
  # load traces
  traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()], dtype=torch.float32)
  # load labels
  ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv(path_to_records, header=None)[0])] # Get order of ids in traces
  df = pd.read_csv(path_to_csv_train)
  df.set_index('id_exam', inplace=True)
  df = df.reindex(ids_traces) # make sure the order is the same
  labels = torch.tensor(np.array(df['AF']), dtype=torch.float32).reshape(-1,1)
  # load dataset
  dataset = TensorDataset(traces, labels)
  len_dataset = len(dataset)
  n_classes = len(torch.unique(labels))

  generator = torch.Generator().manual_seed(seed)
  dataset_train, dataset_valid = random_split(dataset, [0.8, 0.2], generator)

  # build data loaders
  train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
  tqdm.write("Done!\n")
  
  return train_dataloader, valid_dataloader, n_classes, len_dataset