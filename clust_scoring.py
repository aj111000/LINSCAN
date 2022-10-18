import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(X, Y, acc_threshold=.5):
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

    point_acc = C[row_ind, col_ind].sum() / len(Y)

    clust_acc = len([i for i in C[row_ind, col_ind].tolist() if i > acc_threshold]) / len(Y)

    return [point_acc, clust_acc]
