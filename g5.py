from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from datetime import datetime
import time
tree = None
X = None
occurrences = None


def gaussian(x, base, variance):
    return math.exp(-.5 * ((x - base) / variance) ** 2) / variance / math.sqrt(2 * math.pi)


def weight(t, occurrences):
    v = .1
    return sum([gaussian(t, o, v) / (t - o + 1) for o in occurrences])


def distance(v, u, n=1):
    return (abs(v[0] - u[0]) ** n + abs(v[1] - u[1]) ** n) ** float(1 / n)


def fit(_X, _occurrences):
    global X, occurrences, tree
    X = _X
    tree = cKDTree(X)
    occurrences = _occurrences


def query(x, _weight):
    t = time.clock()
    dist, idx = tree.query(x, k=12)
    s = 0
    for i in idx:
        s += _weight(t, occurrences[i])
    s /= distance(x, X[idx[-1]], 2) ** 2 * math.pi
    return s


def main():
    X = np.random.random((10000, 2))
    occurrences = [[] for _ in range(10000)]
    for o in occurrences:
        for _ in range(random.randint(1, 9)):
            o.append(random.randint(1, 1000))
    fit(X, occurrences)
    ss = 0
    nn = 100
    for j in range(nn):
        s = 0
        n = 3000
        for i in range(n):
            x = np.random.random((2))
            # plt.scatter(*x, c='red')
            y = query(x, weight)
            s += y
        ss += s / n
        print(s / n)
        # plt.show()
    print()
    print(ss / nn)


if __name__ == '__main__':
    main()
