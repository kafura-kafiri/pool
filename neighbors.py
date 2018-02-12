from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
tree = None
weights = None
X = None


def distance(v, u):
    return abs(v[0] - u[0]) + abs(v[1] - u[1])


def fit(_X, _weights):
    global X, weights, tree
    X = _X
    print(X)
    tree = cKDTree(X)
    weights = _weights


def query(x):
    t = datetime.now()
    dist, idx = tree.query(x, k=8)
    s = 0
    for i in idx:
        s += weights[i](t) / (distance(x, X[i])) ** 2
    return s / 28


def main():
    X = np.random.random((100, 2))
    w = [lambda x: 1 for i in X]
    fit(X, w)
    plt.scatter(X[:, 0], X[:, 1], c='green')
    s = 0
    n = 20
    for i in range(n):
        x = np.random.random((2))
        plt.scatter(*x, c='red')
        y = query(x)
        s += y
    print(s / n)
    plt.show()


if __name__ == '__main__':
    main()
