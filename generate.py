import argparse
import numpy as np
import tqdm as tqdm
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--number', type=int, default=0)
args = parser.parse_args()

M = 0  # mean value
N = 105  # number of iterations
SIGMA = 25
TAU = 10

C = 0.126
DELTA_X = 1e-3
DELTA_T = 1e-8
L = 1
T = 0.01
Nx = int(L / DELTA_X)
Nt = int(T / DELTA_T)

NUMBER_OF_INITIAL_VALUES = 100

if __name__ == '__main__':
    array_of_u0 = np.zeros((NUMBER_OF_INITIAL_VALUES, Nx))
    array_of_u = np.zeros((NUMBER_OF_INITIAL_VALUES, Nx))
    for j in tqdm.tqdm(range(NUMBER_OF_INITIAL_VALUES)):

        # Generation of u0
        k = np.arange(0, N)
        k = np.array([k])
        k = np.transpose(k)
        x_i = np.linspace(0, L, Nx)
        x_i = np.array([x_i])
        grid = np.matmul(k, x_i) * 2 * np.pi
        A = np.sin(grid) + np.cos(grid)

        a = SIGMA ** 2 * ((2 * k * np.pi) ** 2 + TAU ** 2) ** (-2)

        # for i in range(N):
        b = np.random.normal(size=(N, 1))
        c = np.multiply(np.sqrt(a), b)
        c = np.transpose(c)

        u0 = np.matmul(c, A)

        # Calculating final state
        u = u0
        n = 1  # time
        while n < Nt:
            u = u + DELTA_T * C / DELTA_X ** 2 * (np.roll(u, -1) + np.roll(u, 1) - 2 * u)
            u[-1] = u[0] = (u[-1] + u[0]) / 2
            n += 1

        # Plotting starting value and final state
        # plt.plot(u0[0][:])
        # plt.show()
        # plt.plot(u[0][:])
        # plt.show()

        array_of_u0[j] = u0
        array_of_u[j] = u

    np.save("dummyInitial_" + str(args.number), array_of_u0)
    np.save("dummyTarget_" + str(args.number), array_of_u)

# np.save("test_" + str(args.number), array)
