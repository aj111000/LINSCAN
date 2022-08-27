from data_generation import gen_data
import numpy as np
import numpy.random as random
from multiprocessing import Pool, cpu_count

from run_trials import run_trials
import time
import datetime


def normalize_datasets(datasets):
    for i in range(len(datasets)):
        datasets[i] -= datasets[i].mean(0)

        datasets[i] /= np.max(np.abs(datasets[i]))

        filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])

        datasets[i] = np.array(list(filter(filt, datasets[i].tolist())))
        datasets[i] /= np.max(np.abs(datasets[i]), axis=0)
    return datasets


def param_generator(datasets, labels, eps_range, min_pts_range, threshold_range, ecc_pts_range, xi_range):
    for _ in range(trials):
        eps = gen_rand(eps_range)
        min_pts = int(gen_rand(min_pts_range))
        threshold = gen_rand(threshold_range)
        ecc_pts = int(gen_rand(ecc_pts_range))
        xi = gen_rand(xi_range)
        yield datasets, labels, eps, min_pts, threshold, ecc_pts, xi


if __name__ == '__main__':
    st = time.time()
    # Number of generated datasets and number of validation examples taken each time
    N = 1
    M = 1

    trials = 1

    # Generate Samples
    temp = [gen_data(lin_clusts=6, iso_clusts=3) for i in range(N)]
    train_datasets = [np.array(item[0]) for item in temp]
    train_labels = [np.array(item[1]) for item in temp]

    temp = [gen_data(lin_clusts=6, iso_clusts=3) for i in range(M)]
    test_datasets = [np.array(item[0]) for item in temp]
    test_labels = [np.array(item[1]) for item in temp]
    del temp

    # Normalize Datasets
    x_range = [-1, 1]
    y_range = [-1, 1]

    x_filt = lambda x: x_range[0] <= x <= x_range[1]
    y_filt = lambda y: y_range[0] <= y <= y_range[1]

    train_datasets = normalize_datasets(train_datasets)
    test_datasets = normalize_datasets(test_datasets)

    # Iterations
    eps_range = [.7, .7]
    min_pts_range = [10, 80]
    threshold_range = [.4, .4]
    ecc_pts_range = [10, 80]
    xi_range = [.02, .05]

    scores = []

    gen_rand = lambda range: random.uniform(low=range[0], high=range[1])

    with Pool(processes=1) as pool:
        scores = pool.map(func=run_trials,
                          iterable=param_generator(train_datasets,
                                                   train_labels,
                                                   eps_range,
                                                   min_pts_range,
                                                   threshold_range,
                                                   ecc_pts_range,
                                                   xi_range))

    acc = lambda sample: [np.mean(sample[1]), np.mean(sample[2])]

    point_means = []
    clust_means = []

    for samp in scores:
        samp_point_mean, samp_clust_mean = acc(samp)
        point_means.append(samp_point_mean)
        clust_means.append(samp_clust_mean)

    idx = np.array(point_means).argmax()

    [eps, min_pts, threshold, ecc_pts, xi] = scores[idx][0]

    test_scores = run_trials([test_datasets, test_labels, eps, min_pts, threshold, ecc_pts, xi])

    test_acc = acc(test_scores)
    et = time.time()
    elapsed = et - st
    print("Execution time: ", datetime.timedelta(seconds=elapsed))
