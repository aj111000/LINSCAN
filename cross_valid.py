import multiprocessing

from data_generation import gen_data
import numpy as np
import numpy.random as random
from multiprocessing import Pool, cpu_count

from run_trials import run_trials
import time
import datetime
import sklearn as skl
from clust_scoring import cluster_accuracy


def normalize_datasets(datasets):
    for i in range(len(datasets)):
        datasets[i] -= datasets[i].mean(0)

        datasets[i] /= np.max(np.abs(datasets[i]))

        filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])

        datasets[i] = np.array(list(filter(filt, datasets[i].tolist())))
        datasets[i] /= np.max(np.abs(datasets[i]), axis=0)
    return datasets


def param_generator(datasets, labels, eps_range, min_pts_range, threshold_range, ecc_pts_range, xi_range):
    for i in range(trials):
        eps = gen_rand(eps_range)
        min_pts = int(gen_rand(min_pts_range))
        threshold = gen_rand(threshold_range)
        ecc_pts = int(gen_rand(ecc_pts_range))
        xi = gen_rand(xi_range)
        yield datasets, labels, eps, min_pts, threshold, ecc_pts, xi


if __name__ == '__main__':
    st = time.time()
    # Number of generated datasets and number of validation examples taken each time
    N = 10
    M = 40

    trials = 500

    core_param = multiprocessing.cpu_count() - 1

    # Generate Samples
    temp = [gen_data(lin_clusts=10, iso_clusts=5, int_clusts=10) for i in range(N)]
    train_datasets = [np.array(item[0]) for item in temp]
    train_labels = [np.array(item[1]) for item in temp]

    temp = [gen_data(lin_clusts=10, iso_clusts=5, int_clusts=10) for i in range(M)]
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
    min_pts_range = [20, 80]
    threshold_range = [.2, .5]
    ecc_pts_range = [10, 40]
    xi_range = [.02, .06]

    scores = []

    gen_rand = lambda range: random.uniform(low=range[0], high=range[1])

    with Pool(processes=min(trials, cpu_count(), core_param)) as pool:
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
    print("LINSCAN:\n")
    print(scores[idx][0])
    print([scores[idx][1][0], scores[idx][2][0]])
    print(test_acc)


    # OPTICS
    min_pts_range = [10, 100]
    threshold_range = [.3, .6]
    xi_range = [.02, .06]

    optics_scores = []

    for _ in range(trials):
        point_scores = []
        clust_scores = []
        min_pts = int(np.round(gen_rand(min_pts_range)))
        threshold = gen_rand(threshold_range)
        xi = gen_rand(xi_range)

        optics_classifier = skl.cluster.OPTICS(min_samples=min_pts, xi=xi)

        for dataset, true_label in zip(train_datasets, train_labels):
            label = optics_classifier.fit_predict(dataset)

            for cat in range(max(label)):
                temp = np.array([dataset[i, :] for i in range(len(dataset)) if label[i] == cat])
                if temp.size == 0:
                    continue
                if np.abs(np.corrcoef(temp, rowvar=False)[0, 1]) < threshold:
                    label = list(map(lambda x: -1 if x == cat else x, label))

            X = []

            for i in range(max(label) + 1):
                X.append({idx for idx in range(len(label)) if label[idx] == i})

            Y = []

            for i in range(max(true_label) + 1):
                Y.append({idx for idx in range(len(true_label)) if true_label[idx] == i})

            point_acc, clust_acc = cluster_accuracy(X, Y)
            point_scores.append(point_acc)
            clust_scores.append(clust_acc)

        optics_scores.append([[min_pts, threshold], point_scores, clust_scores])

    optics_point_means = []
    optics_clust_means = []

    for samp in optics_scores:
        samp_point_mean, samp_clust_mean = acc(samp)
        optics_point_means.append(samp_point_mean)
        optics_clust_means.append(samp_clust_mean)

    optics_idx = np.array(optics_point_means).argmax()

    [min_pts, threshold] = optics_scores[optics_idx][0]

    optics_classifier = skl.cluster.OPTICS(min_samples=min_pts)

    optics_test_scores = []

    for dataset, true_label in zip(test_datasets, test_labels):
        label = optics_classifier.fit_predict(dataset)

        for cat in range(max(label)):
            temp = np.array([dataset[i, :] for i in range(len(dataset)) if label[i] == cat])
            if temp.size == 0:
                continue
            if np.abs(np.corrcoef(temp, rowvar=False)[0, 1]) < threshold:
                label = list(map(lambda x: -1 if x == cat else x, label))

        X = []

        for i in range(max(label) + 1):
            X.append({idx for idx in range(len(label)) if label[idx] == i})

        Y = []

        for i in range(max(true_label) + 1):
            Y.append({idx for idx in range(len(true_label)) if true_label[idx] == i})

        point_acc, clust_acc = cluster_accuracy(X, Y)
        point_scores.append(point_acc)
        clust_scores.append(clust_acc)

    optics_test_scores = [[min_pts, threshold], point_scores, clust_scores]

    optics_test_acc = acc(optics_test_scores)
    print("\nOptics:\n")
    print(optics_scores[optics_idx][0])
    print([optics_scores[optics_idx][1][0], optics_scores[optics_idx][2][0]])
    print(optics_test_acc)
