from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
tree = None
weights = None
X = None


def gaussian(x, base, variance):
    return math.exp(-.5 * ((x - base) / variance) ** 2) / variance / math.sqrt(2 * math.pi)


def weight(t, occurrences):
    v = .1
    return sum([gaussian(t, o, v) / (t - o + 1) for o in occurrences])


def distance(v, u, n=1):
    return (abs(v[0] - u[0]) ** n + abs(v[1] - u[1]) ** n) ** float(1 / n)


def fit(_X, _weights):
    global X, weights, tree
    X = _X
    print(X)
    tree = cKDTree(X)
    weights = _weights


def query(x):
    t = datetime.now()
    dist, idx = tree.query(x, k=12)
    s = 0
    for i in idx:
        s += weights[i](t)
    s /= distance(x, X[idx[-1]], 2) ** 2 * math.pi
    return s


def main():
    X = np.random.random((10000, 2))
    w = [lambda t: 1 for _ in X]
    fit(X, w)
    ss = 0
    nn = 100
    for j in range(nn):
        s = 0
        n = 3000
        for i in range(n):
            x = np.random.random((2))
            # plt.scatter(*x, c='red')
            y = query(x)
            s += y
        ss += s / n
        print(s / n)
        # plt.show()
    print()
    print(ss / nn)


if __name__ == '__main__':
    main()
