from lin_scan import linscan

# from clust_scoring import cluster_accuracy
from sklearn import metrics

def run_trials(args):
    datasets, true_labels, eps, min_pts, threshold, ecc_pts, xi = args
    scores = []
    # point_scores = []
    # clust_scores = []
    for (dataset, true_label) in zip(datasets, true_labels):
        gen_label = linscan(dataset, eps, min_pts, ecc_pts, threshold, xi)

        scores.append(metrics.adjusted_rand_score(true_label, gen_label))
        # X = []
        #
        # for i in range(max(gen_label) + 1):
        #     X.append({idx for idx in range(len(gen_label)) if gen_label[idx] == i})
        #
        # Y = []
        #
        # for i in range(max(true_label) + 1):
        #     Y.append({idx for idx in range(len(true_label)) if true_label[idx] == i})
        #
        # point_acc, clust_acc = cluster_accuracy(X, Y)
        # point_scores.append(point_acc)
        # clust_scores.append(clust_acc)

    print("Done")
    return [[eps, min_pts, threshold, ecc_pts, xi], scores]
    # return [[eps, min_pts, threshold, ecc_pts, xi], point_scores, clust_scores]
