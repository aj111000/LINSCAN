import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
# from scipy.linalg import sqrtm as sqrtm
import sklearn.neighbors as neighbors
from scipy.spatial.distance import jensenshannon

from sklearn.cluster import OPTICS

from import_coordinates import import_test, import_hs

def sqrtm(mat):
    s = np.sqrt(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return 1 / t * (mat + s * np.eye(2))

def kl_dist(x, y):
    # (x_1, x_2, s_11, s_12, s_22)
    cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
    inv1 = 1 / (x[2] * x[4] - x[3] * x[3]) * np.array([[x[4], -x[3]], [-x[3], x[2]]])
    invsqrt1 = sqrtm(inv1)
    p1 = np.array([x[0], x[1]])

    cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
    inv2 = 1 / (y[2] * y[4] - y[3] * y[3]) * np.array([[y[4], -y[3]], [-y[3], y[2]]])
    invsqrt2 = sqrtm(inv2)
    p2 = np.array([y[0], y[1]])

    dist = 1 / 2 * np.sqrt(np.linalg.norm(invsqrt2 @ cov1 @ invsqrt2 - np.eye(2), ord='fro')) \
           + 1 / 2 * np.sqrt(np.linalg.norm(invsqrt1 @ cov2 @ invsqrt1 - np.eye(2), ord='fro')) \
           + 1 / np.sqrt(2) * np.sqrt((p1 - p2).transpose() @ inv1 @ (p1 - p2)) \
           + 1 / np.sqrt(2) * np.sqrt((p1 - p2).transpose() @ inv2 @ (p1 - p2))

    return np.max([dist, 0])


def kl_embed_scan(dataset, eps, min_pts, ecc_pts, xi=.05):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
        cov /= max(np.linalg.eig(cov)[0])
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

        embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
    embeddings = np.array(embeddings)

    return OPTICS(min_samples=min_pts, eps=eps, metric=kl_dist, cluster_method="xi", xi=xi).fit(embeddings)


def linscan(dataset, eps, min_pts, ecc_pts, threshold, xi):

    optics = kl_embed_scan(dataset, eps, min_pts, ecc_pts, xi)

    typelist = optics.labels_

    for cat in range(max(typelist)):
        temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] == cat])
        if temp.size == 0:
            continue
        if np.abs(np.corrcoef(temp, rowvar=False)[0, 1]) < threshold:
            typelist = list(map(lambda x: -1 if x == cat else x, typelist))

    return typelist

# if __name__ == '__main__':
#     np.seterr(all='raise')
#     from lin_scan_old import swiss_roll, crossing_lines
#
#     # read data
#     dataset = np.array(import_test())
#
#     dataset -= dataset.mean(0)
#
#     dataset /= np.max(np.abs(dataset))
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.set_aspect('equal', adjustable='box')
#     #
#     # plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', s=(2*72./fig.dpi)**2)
#     # plt.show()
#
#     x_range = [-1, 1]
#     y_range = [-1, 1]
#
#     x_filt = lambda x: x_range[0] <= x <= x_range[1]
#     y_filt = lambda y: y_range[0] <= y <= y_range[1]
#
#     filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])
#
#     dataset = np.array(list(filter(filt, dataset.tolist())))
#     dataset /= np.max(np.abs(dataset), axis=0)
#
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.set_aspect('equal', adjustable='box')
#     #
#     # plt.scatter(dataset[:, 0], dataset[:, 1], marker='o', s=(2 * 72. / fig.dpi) ** 2)
#     # plt.show()
#
#     eps = np.inf
#     min_pts = 60
#     threshold = 0
#     ecc_pts = 30
#     xi = .02
#
#     multiplier = 1
#
#     optics = kl_embed_scan(dataset, eps, min_pts, ecc_pts, xi)
#     space = np.arange(len(dataset))
#     reachability = optics.reachability_[optics.ordering_]
#     labels = optics.labels_[optics.ordering_]
#
#     typelist = optics.labels_
#
#     fig1 = plt.figure()
#     ax = fig1.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')
#
#     plt.scatter(dataset[:, 0], dataset[:, 1], c=typelist, marker='o', s=(2 * 72. / fig1.dpi) ** 2)
#
#     plt.show()
#
#     temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] != -1])
#     temp_list = np.array([typelist[i] for i in range(len(typelist)) if typelist[i] != -1])
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     plt.scatter(space, reachability, c=typelist[optics.ordering_], marker='.', s=(2 * 72. / fig.dpi) ** 2)
#     plt.show()
#
#     fig2 = plt.figure()
#     ax = fig2.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')
#
#     plt.scatter(temp[:, 0], temp[:, 1], c=temp_list, marker='o', s=(2 * 72. / fig.dpi) ** 2)
#     plt.show()
#
#     for cat in range(max(typelist)):
#         temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] == cat])
#         if temp.size == 0:
#             continue
#         if np.abs(np.corrcoef(temp, rowvar=False)[0, 1]) < threshold:
#             typelist = list(map(lambda x: -1 if x == cat else x, typelist))
#
#     temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] != -1])
#     temp_list = np.array([typelist[i] for i in range(len(typelist)) if typelist[i] != -1])
#
#     fig3 = plt.figure()
#     ax = fig3.add_subplot(111)
#     ax.set_aspect('equal', adjustable='box')
#
#     plt.scatter(temp[:, 0], temp[:, 1], c=temp_list, marker='o', s=(2 * 72. / fig.dpi) ** 2)
#     plt.show()
