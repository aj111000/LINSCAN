from data_generation import gen_data
from clust_scoring import cluster_accuracy
from lin_scan import linscan
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt


def run_trials(inds, datasets, true_labels, eps, min_pts, ecc_pts, threshold, xi):
    scores = []
    for ind in inds:
        dataset = datasets[ind]
        true_label = true_labels[ind]

        gen_label = linscan(dataset, eps, min_pts, ecc_pts, threshold, xi)

        X = []

        for i in range(max(true_label) + 1):
            X.append({idx for idx in range(len(true_label)) if true_label[idx] == i})

        Y = []

        for i in range(max(gen_label) + 1):
            Y.append({idx for idx in range(len(gen_label)) if gen_label[idx] == i})

        acc = cluster_accuracy(X, Y)
        scores.append(acc)

    return scores


# Number of generated datasets and number of validation examples taken each time
N = 6
k = 2

# Generate Samples
temp = [gen_data() for i in range(N)]
datasets = [np.array(item[0]) for item in temp]
true_labels = [np.array(item[1]) for item in temp]
del temp

# Normalize Datasets
x_range = [-1, 1]
y_range = [-1, 1]

x_filt = lambda x: x_range[0] <= x <= x_range[1]
y_filt = lambda y: y_range[0] <= y <= y_range[1]

for i in range(N):
    datasets[i] -= datasets[i].mean(0)

    datasets[i] /= np.max(np.abs(datasets[i]))

    filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])

    datasets[i] = np.array(list(filter(filt, datasets[i].tolist())))
    datasets[i] /= np.max(np.abs(datasets[i]), axis=0)

# Iterations
eps_list = [10]
min_pts_list = [30, 70, 120]
threshold_list = [.3, .6]
ecc_pts_list = [30, 70, 120]
xi_list = [.03, .05]

val_idx = 0

scores = []

for [eps, min_pts, threshold, ecc_pts, xi] in iter.product(
        eps_list, min_pts_list, threshold_list, ecc_pts_list, xi_list
):
    iter_scores = [[eps, min_pts, threshold, ecc_pts, xi]]

    val_ind = [(val_idx + i) % N for i in range(k)]
    train_ind = [(val_idx + k + i) % N for i in range(N - k)]

    train_scores = run_trials(train_ind, datasets, true_labels, eps, min_pts, ecc_pts, threshold, xi)
    val_scores = run_trials(val_ind, datasets, true_labels, eps, min_pts, ecc_pts, threshold, xi)

    iter_scores += [train_scores, val_scores]

    scores.append(iter_scores)

    val_idx = (val_idx + k) % N
