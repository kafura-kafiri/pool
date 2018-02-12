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
    x = x - base
    h = x / 3600
    y = math.exp(-.5 * ((h % 24) / variance) ** 2) / variance / math.sqrt(2 * math.pi) / ( h + 1 )
    return y


def weight(t, occurrences):
    v = .4
    return sum([gaussian(t, o, v) / (t - o + 1) for o in occurrences])


def distance(v, u, n=1):
    return (abs(v[0] - u[0]) ** n + abs(v[1] - u[1]) ** n) ** float(1 / n)


def fit(_X, _occurrences):
    global X, occurrences, tree
    X = _X
    tree = cKDTree(X)
    occurrences = _occurrences


def query(x, _weight):
    t = time.time()
    dist, idx = tree.query(x, k=12)
    s = 0
    for i in idx:
        s += _weight(t, occurrences[i])
    s /= distance(x, X[idx[-1]], 2) ** 2 * math.pi
    return s


def main():
    n = 20
    X = np.random.random((n, 2))
    t = time.time()
    occurrences = [[] for _ in range(n)]
    for o in occurrences:
        for _ in range(random.randint(1, 9)):
            o.append(t - random.randint(1, 10000))
    fit(X, occurrences)
    s = 0
    n = 1
    for i in range(n):
        x = np.random.random((2))
        # plt.scatter(*x, c='red')
        y = query(x, weight)
        s += y
    print(s / n)


if __name__ == '__main__':
    main()
