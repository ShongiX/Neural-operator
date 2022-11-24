import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

initial = np.load("data/dummyInitial_0.npy")
target = np.load("data/dummyTarget_0.npy")

tensor_x = torch.Tensor(initial)  # transform to torch tensor
tensor_y = torch.Tensor(target)

my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
my_dataloader = DataLoader(my_dataset)  # create your dataloader
