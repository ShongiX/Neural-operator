import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

initial = np.load("data/dummyInitial_0.npy")
target = np.load("data/dummyTarget_0.npy")

for number in range(9):
    initial = np.concatenate(initial, np.load("data/dummyInitial_" + str(number+1) + ".npy"))
    target = np.concatenate(target, np.load("data/dummyTarget_" + str(number+1) + ".npy"))

tensorInitial = torch.Tensor(initial)
tensorTarget = torch.Tensor(target)

HeatConductionDataset = TensorDataset(tensorInitial, tensorTarget)
HeadConductionDataLoader = DataLoader(HeatConductionDataset)

