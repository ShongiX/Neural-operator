import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import time

k_max = 12
width = 64
J = 1000  # number of points in the domain
batch_size = 20


class FourierLayer(nn.Module):
    def __init__(self, size_in, size_out, modes):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.modes = modes

        scale = 1 / (size_in * size_out)
        self.R = nn.Parameter(scale * torch.randn(self.size_in, self.size_out, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchSize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        y = torch.zeros(batchSize, self.size_out, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        y[:, :, :self.modes] = torch.einsum("bnk,nmk->bmk", x_ft[:, :, :self.modes], self.R)
        x = torch.fft.irfft(y)
        return x


class HeatConductionNeuralNetwork(nn.Module):
    def __init__(self, modes, width):
        super(HeatConductionNeuralNetwork, self).__init__()

        self.modes = modes
        self.width = width

        self.P = nn.Linear(1, self.width)  # uplift layer

        self.FL1 = FourierLayer(self.width, self.width, self.modes)
        self.FL2 = FourierLayer(self.width, self.width, self.modes)
        self.FL3 = FourierLayer(self.width, self.width, self.modes)
        self.FL4 = FourierLayer(self.width, self.width, self.modes)

        self.W1 = nn.Conv1d(self.width, self.width, 1)
        self.W2 = nn.Conv1d(self.width, self.width, 1)
        self.W3 = nn.Conv1d(self.width, self.width, 1)
        self.W4 = nn.Conv1d(self.width, self.width, 1)

        self.Q = nn.Linear(self.width, 1)  # projection layer

    def forward(self, x):
        x = self.P(x)
        x = torch.permute(x, (0, 2, 1))

        v1 = self.FL1(x)
        v2 = self.W1(x)
        x = nn.functional.relu(v1 + v2)

        v1 = self.FL2(x)
        v2 = self.W2(x)
        x = nn.functional.relu(v1 + v2)

        v1 = self.FL3(x)
        v2 = self.W3(x)
        x = nn.functional.relu(v1 + v2)

        v1 = self.FL4(x)
        v2 = self.W4(x)
        x = nn.functional.relu(v1 + v2)

        x = torch.permute(x, (0, 2, 1))
        x = self.Q(x)
        return x


def main():
    start_time = time.time()

    model = HeatConductionNeuralNetwork(modes=k_max, width=width)
    model.load_state_dict(torch.load("model.pth"))

    model.eval()

    initial_test = np.load("data/dummyInitial_9.npy")
    target_test = np.load("data/dummyTarget_9.npy")
    initial_test = initial_test.reshape(400, J, 1)

    init = torch.Tensor(initial_test)
    y = torch.Tensor(target_test)

    with torch.no_grad():
        pred = model(init)
        pred = np.squeeze(pred)

        x = np.linspace(0, 1, 1000)
        plt.plot(x, y[0, :], 'g')
        plt.plot(x, pred[0, :], 'b')
        plt.title("Prediction")
        plt.xlabel("Position along rod")
        plt.ylabel("Temperature")
        plt.show()

    print("Done! (%s)" % (time.time() - start_time))


if __name__ == "__main__":
    main()
