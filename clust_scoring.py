import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(X, Y):
    # X and Y are lists of sets corresponding to the points in each cluster

    C = []
    for x in X:
        c = []
        for y in Y:
            total = x.union(y)
            intersect = x.intersection(y)
            c.append(len(intersect) / len(total))

        C.append(c)

    C = np.array(C)
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / min(len(X), len(Y))
