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

initial = np.load("data/dummyInitial_0.npy")
target = np.load("data/dummyTarget_0.npy")

for number in range(7):
    initial = np.concatenate((initial, np.load("data/dummyInitial_" + str(number + 1) + ".npy")))
    target = np.concatenate((target, np.load("data/dummyTarget_" + str(number + 1) + ".npy")))

initial = initial.reshape(3200, J, 1)

HeatConductionTrainingDataset = TensorDataset(torch.Tensor(initial), torch.Tensor(target))
HeatConductionTrainingDataLoader = DataLoader(HeatConductionTrainingDataset, batch_size=batch_size, shuffle=True)

initial = np.load("data/dummyInitial_8.npy")
target = np.load("data/dummyTarget_8.npy")

initial = initial.reshape(400, J, 1)

HeatConductionValidationDataset = TensorDataset(torch.Tensor(initial), torch.Tensor(target))
HeatConductionValidationDataLoader = DataLoader(HeatConductionValidationDataset, batch_size=batch_size)


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


train_loss = []


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        pred = np.squeeze(pred)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_loss.append(loss)


validation_loss = []


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = np.squeeze(pred)
            test_loss += loss_fn(pred, y).item()

            x = np.linspace(0, 1, 1000)
            plt.plot(x, y[0, :], 'g')
            plt.plot(x, pred[0, :], 'b')
            plt.title("Prediction")
            plt.xlabel("Position along rod")
            plt.ylabel("Temperature")
            plt.show()

            correct += np.count_nonzero(((np.trapz(pred) - np.trapz(y)) < 0.05))

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    validation_loss.append(test_loss)


def main():
    start_time = time.time()
    model = HeatConductionNeuralNetwork(modes=k_max, width=width)

    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(HeatConductionTrainingDataLoader, model, loss_fn, optimizer)
        test_loop(HeatConductionValidationDataLoader, model, loss_fn)
    print("Done! (%s)" % (time.time() - start_time))

    # plt.plot(np.linspace(1, 10, 10), validation_loss)
    # plt.plot(validation_loss)
    # plt.title("Average validation loss")
    # plt.show()
    #
    # plt.plot(np.linspace(1, 40, 40), train_loss)
    # plt.title("Training loss")
    # plt.show()
    #
    # plt.plot(np.linspace(1, 20, 20), train_loss[20:])
    # plt.title("Training loss")
    # plt.show()

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
