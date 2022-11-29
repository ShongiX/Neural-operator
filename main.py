import math

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

k_max = 12
UPLIFT = 1000
J = 1000  # number of points in the domain
batch_size = 20


initial = np.load("data/dummyInitial_0.npy")
target = np.load("data/dummyTarget_0.npy")

for number in range(8):
    initial = np.concatenate((initial, np.load("data/dummyInitial_" + str(number + 1) + ".npy")))
    target = np.concatenate((target, np.load("data/dummyTarget_" + str(number + 1) + ".npy")))

HeatConductionTrainingDataset = TensorDataset(torch.Tensor(initial), torch.Tensor(target))
HeatConductionTrainingDataLoader = DataLoader(HeatConductionTrainingDataset, batch_size=batch_size, shuffle=True)

initial = np.load("data/dummyInitial_8.npy")
target = np.load("data/dummyTarget_8.npy")

HeatConductionValidationDataset = TensorDataset(torch.Tensor(initial), torch.Tensor(target))
HeatConductionValidationDataLoader = DataLoader(HeatConductionValidationDataset, batch_size=batch_size)


class FourierLayer(nn.Module):
    def __init__(self, size_in, size_out, modes):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.modes = modes

        self.R = nn.Parameter(torch.rand(modes, size_in, size_out, dtype=torch.cfloat))

    def forward(self, x):
        batchSize = x.shape[0]
        x_ft = torch.fft.fft(x)
        print(x_ft.shape)
        y = torch.zeros(batchSize, self.size_out, self.modes, device=x.device, dtype=torch.cfloat)
        y[:, :, :self.modes] = torch.einsum("kmn,bn->bmk", self.R[:self.modes, :, :], x_ft)
        x = torch.fft.ifft(y)
        return x


class HeatConductionNeuralNetwork(nn.Module):
    def __init__(self, modes, uplift):
        super(HeatConductionNeuralNetwork, self).__init__()

        self.modes = modes
        self.uplift = uplift

        self.P = nn.Linear(J, uplift)  # uplift layer

        self.FL1 = FourierLayer(self.uplift, self.uplift, modes)
        self.FL2 = FourierLayer(self.uplift, self.uplift, modes)
        self.FL3 = FourierLayer(self.uplift, self.uplift, modes)
        self.FL4 = FourierLayer(self.uplift, self.uplift, modes)

        self.W1 = nn.Conv1d(self.uplift, self.uplift, 1)
        self.W2 = nn.Conv1d(self.uplift, self.uplift, 1)
        self.W3 = nn.Conv1d(self.uplift, self.uplift, 1)
        self.W4 = nn.Conv1d(self.uplift, self.uplift, 1)

        self.Q = nn.Linear(uplift, J)  # lowering layer

    def forward(self, x):
        x = self.P(x)
        v1 = self.FL1(x)
        v2 = self.W1(x)
        x = nn.functional.relu(v1+v2)

        v1 = self.FL2(x)
        v2 = self.W2(x)
        x = nn.functional.relu(v1 + v2)

        v1 = self.FL3(x)
        v2 = self.W3(x)
        x = nn.functional.relu(v1 + v2)

        v1 = self.FL4(x)
        v2 = self.W4(x)
        x = nn.functional.relu(v1 + v2)

        x = self.Q(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    model = HeatConductionNeuralNetwork(modes=k_max, uplift=UPLIFT)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(HeatConductionTrainingDataLoader, model, loss_fn, optimizer)
        test_loop(HeatConductionValidationDataLoader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
