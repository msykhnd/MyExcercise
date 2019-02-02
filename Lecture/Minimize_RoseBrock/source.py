import numpy as np


def RosenBrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2)


if __name__ == "__main__":
    x = np.ones(50)
    # print(1-x[:-1])
    # x = np.arange(0, 50)
    print(x)
    print(RosenBrock(x))
