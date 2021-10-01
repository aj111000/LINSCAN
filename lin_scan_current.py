import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
# from scipy.linalg import sqrtm as sqrtm
import sklearn.neighbors as neighbors

from import_coordinates import import_test, import_hs


class VisitList:
    def __init__(self, count):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitednum = count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1


def sqrtm(mat):
    s = np.sqrt(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return 1 / t * (mat + s * np.eye(2))


def kl_dist(x, y):
    # (x_1, x_2, s_11, s_12, s_22)
    cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
    inv1 = np.linalg.inv(cov1)
    p1 = np.array([x[0], x[1]])

    cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
    inv2 = np.linalg.inv(cov2)
    p2 = np.array([y[0], y[1]])

    return np.sqrt(
        np.max([(p1 - p2).transpose() @ (inv1 + inv2) @ (p1 - p2) + np.trace(inv1 @ cov2 + inv2 @ cov1) - 4, 0]))


def mmd_dist(x, y):
    c1 = []
    c2 = []

    if len(x) != len(y):
        raise "Not the same length"

    for i in range(int(len(x) / 2)):
        c1.append(np.array([x[2 * (i - 1)], x[2 * i - 1]]))
        c2.append(np.array([y[2 * (i - 1)], y[2 * i - 1]]))

    k = lambda a, b: np.exp(-np.linalg.norm(a - b) ** 2 / 2)

    n = len(c1)
    MMD1 = 0
    MMD2 = 0
    MMD3 = 0
    for a in c1:
        for b in c1:
            MMD1 += k(a, b)

        for b in c2:
            MMD2 += 2 * k(a, b)

    for a in c2:
        for b in c2:
            MMD3 += k(a, b)

    return MMD1 / (n * (n - 1)) - MMD2 / n ** 2 + MMD3 / (n * (n - 1))


def was_wrapper(multiplier):
    def was_dist(x, y):
        # (x_1, x_2, s_11, s_12, s_22)
        cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
        p1 = np.array([x[0], x[1]])
        sqrt1 = sqrtm(cov1)

        cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
        p2 = np.array([y[0], y[1]])

        temp_mat = sqrt1 @ cov2 @ sqrt1

        return np.sqrt(
            np.max([np.linalg.norm(p1 - p2) ** 2 + multiplier * np.trace(cov1 + cov2 - 2 * sqrtm(temp_mat)), 0]))

    return was_dist


def db_scan(dataset, eps, min_pts):
    nPoints = dataset.shape[0]
    vPoints = VisitList(nPoints)
    current_cat = -1
    categories = [-1 for i in range(nPoints)]
    kd = KDTree(dataset)

    while vPoints.unvisitednum > 0:
        p = random.choice(vPoints.unvisitedlist)
        cluster = kd.query_ball_point(x=dataset[p], r=eps)
        vPoints.visit(p)

        if len(cluster) >= min_pts:
            current_cat += 1
            categories[p] = current_cat
            while len(cluster) > 0:
                p1 = cluster.pop(0)

                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    cluster1 = kd.query_ball_point(x=dataset[p1], r=eps)
                    if len(cluster1) >= min_pts:
                        categories[p1] = current_cat
                        cluster = cluster + cluster1

        if categories.count(current_cat) <= min_pts:
            for i in range(len(categories)):
                if categories[i] == current_cat:
                    categories[i] = -1
            current_cat -= 1
        else:
            categories[p] = -1
        print(len(vPoints.unvisitedlist))
    return categories


def kl_db_scan(dataset, eps, min_pts):
    nPoints = dataset.shape[0]
    vPoints = VisitList(nPoints)
    current_cat = -1
    categories = [-1 for i in range(nPoints)]
    kd = BallTree(dataset, metric="pyfunc", func=kl_dist)

    while vPoints.unvisitednum > 0:
        p = random.choice(vPoints.unvisitedlist)
        cluster = kd.query_radius(X=[dataset[p]], r=eps)[0].tolist()
        vPoints.visit(p)

        if len(cluster) >= min_pts:
            current_cat += 1
            categories[p] = current_cat
            while len(cluster) > 0:
                p1 = cluster.pop(0)

                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    cluster1 = kd.query_radius(X=[dataset[p1]], r=eps)[0].tolist()
                    if len(cluster1) >= min_pts:
                        categories[p1] = current_cat
                        cluster = cluster + cluster1

        if categories.count(current_cat) <= min_pts:
            for i in range(len(categories)):
                if categories[i] == current_cat:
                    categories[i] = -1
            current_cat -= 1
        else:
            categories[p] = -1
        print(len(vPoints.unvisitedlist))
    return categories


def was_db_scan(dataset, eps, min_pts, multiplier):
    nPoints = dataset.shape[0]
    vPoints = VisitList(nPoints)
    current_cat = -1
    categories = [-1 for i in range(nPoints)]
    kd = BallTree(dataset, metric="pyfunc", func=was_wrapper(multiplier))

    while vPoints.unvisitednum > 0:
        p = random.choice(vPoints.unvisitedlist)
        cluster = kd.query_radius(X=[dataset[p]], r=eps)[0].tolist()
        vPoints.visit(p)

        if len(cluster) >= min_pts:
            current_cat += 1
            categories[p] = current_cat
            while len(cluster) > 0:
                p1 = cluster.pop(0)

                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    cluster1 = kd.query_radius(X=[dataset[p1]], r=eps)[0].tolist()
                    if len(cluster1) >= min_pts:
                        categories[p1] = current_cat
                        cluster = cluster + cluster1

        if categories.count(current_cat) <= min_pts:
            for i in range(len(categories)):
                if categories[i] == current_cat:
                    categories[i] = -1
            current_cat -= 1
        else:
            categories[p] = -1
        print(len(vPoints.unvisitedlist))
    return categories


def mmd_db_scan(dataset, eps, min_pts):
    nPoints = dataset.shape[0]
    vPoints = VisitList(nPoints)
    current_cat = -1
    categories = [-1 for i in range(nPoints)]
    kd = BallTree(dataset, metric="pyfunc", func=mmd_dist)

    while vPoints.unvisitednum > 0:
        p = random.choice(vPoints.unvisitedlist)
        cluster = kd.query_radius(X=[dataset[p]], r=eps)[0].tolist()
        vPoints.visit(p)

        if len(cluster) >= min_pts:
            current_cat += 1
            categories[p] = current_cat
            while len(cluster) > 0:
                p1 = cluster.pop(0)

                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    cluster1 = kd.query_radius(X=[dataset[p1]], r=eps)[0].tolist()
                    if len(cluster1) >= min_pts:
                        categories[p1] = current_cat
                        cluster = cluster + cluster1

        if categories.count(current_cat) <= min_pts:
            for i in range(len(categories)):
                if categories[i] == current_cat:
                    categories[i] = -1
            current_cat -= 1
        else:
            categories[p] = -1
        print(len(vPoints.unvisitedlist))
    return categories


def calc_corr(cluster):
    return np.corrcoef(np.array(cluster), rowvar=False)[0, 1]


def euc_embed_scan(dataset, eps, min_pts, ecc_pts):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = (np.cov(np.array([dataset[k] for k in cluster]), rowvar=False))
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

        embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))

    return db_scan(dataset, eps, min_pts)


def kl_embed_scan(dataset, eps, min_pts, ecc_pts):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

        embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
    embeddings = np.array(embeddings)

    return kl_db_scan(embeddings, eps, min_pts)


def was_embed_scan(dataset, eps, min_pts, ecc_pts, multiplier):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

        embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
    embeddings = np.array(embeddings)

    return was_db_scan(embeddings, eps, min_pts, multiplier)


def mmd_embed_scan(dataset, eps, min_pts, ecc_pts):
    from functools import reduce
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cluster = [dataset[k].tolist() for k in cluster]
        cluster = reduce(lambda x, y: x + y, cluster)
        embeddings.append(cluster)
    embeddings = np.array(embeddings)

    return mmd_db_scan(embeddings, eps, min_pts)


# hist = [kl_dist(embeddings[p],embeddings[q]) for p in range(len(embeddings)) for q in range(len(embeddings))]

# def lin_scan(dataset, eps, min_pts, ecc_pts):
#     nPoints = dataset.shape[0]
#     vPoints = VisitList(nPoints)
#     current_cat = -1
#     categories = [-1 for i in range(nPoints)]
#     kd = KDTree(dataset)
#
#     def neighborhood(p, ecc_pts, dataset):
#         cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
#         return [dataset[k] for k in cluster]
#
#     correlations = [calc_corr(neighborhood(p, ecc_pts, dataset)) for p in vPoints.unvisitedlist]
#     vPoints.unvisitedlist = [x for _, x in sorted(zip(correlations, vPoints.unvisitedlist), key=lambda pair: pair[0])]
#
#     mah_dist = lambda x, y, inv_cov: np.sqrt((x - y).transpose() @ inv_cov @ (x - y))
#
#     while vPoints.unvisitednum > 0:
#         p = vPoints.unvisitedlist[0]
#         cluster = set(kd.query(x=dataset[p], k=ecc_pts)[1].tolist())
#         N = len(cluster)
#         vPoints.visit(p)
#
#         if len(cluster) >= min_pts:
#             cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
#             inv_cov = np.linalg.inv(cov)
#
#             mean = np.mean([dataset[k] for k in cluster], axis=0)
#
#             current_cat += 1
#             categories[p] = current_cat
#             while len(cluster) > 0:
#                 p1 = cluster.pop()
#
#                 if p1 in vPoints.unvisitedlist:
#
#                     cluster1 = [k for k in range(len(dataset)) if mah_dist(dataset[p1], dataset[k], inv_cov) < eps]
#
#                     if len(cluster1) >= min_pts:
#                         categories[p1] = current_cat
#                         vPoints.visit(p1)
#                         N += 1
#
#                         X = dataset[p1]
#
#                         cov = (N-2)/(N-1) * cov + (X-mean)@(X-mean).transpose()
#
#                         mean = (N-1)/N * mean + 1/N * X
#
#                         cluster.update(cluster1)
#
#         if categories.count(current_cat) <= min_pts:
#             for i in range(len(categories)):
#                 if categories[i] == current_cat:
#                     categories[i] = -1
#             current_cat -= 1
#         else:
#             categories[p] = -1
#         print(len(vPoints.unvisitedlist))
#     return categories

# if __name__ == '__main__':
#     np.seterr(all='raise')
#     from lin_scan_old import swiss_roll, crossing_lines
#
#     # read data
#     dataset = np.array(import_hs())
#
#     dataset /= np.max(np.abs(dataset))
#
#     x_range = [-1, 1]
#     y_range = [-1, 0]
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
#     # plt.scatter(dataset[:, 0], dataset[:, 1], s=80, facecolors='none', edgecolors='b')
#     # plt.show()
#
#     ecc_pts = 15
#
#     dists = []
#
#     kd = KDTree(dataset)
#
#     embeddings = []
#     for p in range(len(dataset)):
#         cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
#         cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
#         mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)
#
#         embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
#     embeddings = np.array(embeddings)
#
#     x = []
#     y = []
#     trials = 1000
#
#     while trials > 0:
#         [i, j, k] = random.sample(range(len(embeddings)), 3)
#         i = embeddings[i]
#         j = embeddings[j]
#         k = embeddings[k]
#
#         temp_x = kl_dist(i, j)
#         temp_y = kl_dist(j, k)
#         temp_z = kl_dist(i, k)
#
#         if max([temp_x, temp_y, temp_z]) < np.sqrt(40):
#             y.append(temp_x)
#             x.append(temp_y+temp_z)
#
#             trials -= 1
#
#     plt.scatter(x=x, y=y)
#     plt.scatter(x=x, y=x)
#     plt.show()

if __name__ == '__main__':
    np.seterr(all='raise')
    from lin_scan_old import swiss_roll, crossing_lines

    # read data
    dataset = np.array(import_hs())

    dataset -= dataset.mean(0)

    dataset /= np.max(np.abs(dataset))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(dataset[:, 0], dataset[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.show()

    x_range = [-1, 1]
    y_range = [-1, -.5]

    x_filt = lambda x: x_range[0] <= x <= x_range[1]
    y_filt = lambda y: y_range[0] <= y <= y_range[1]

    filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])

    dataset = np.array(list(filter(filt, dataset.tolist())))
    dataset /= np.max(np.abs(dataset), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(dataset[:, 0], dataset[:, 1], marker='.')
    plt.show()

    eps = np.sqrt(9)
    min_pts = 10
    threshold = .4
    ecc_pts = 15

    multiplier = 1

    typelist = kl_embed_scan(dataset, eps, min_pts, ecc_pts)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(dataset[:, 0], dataset[:, 1], c=typelist, marker='.')

    plt.show()

    temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] != -1])
    temp_list = np.array([typelist[i] for i in range(len(typelist)) if typelist[i] != -1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(temp[:, 0], temp[:, 1], c=temp_list, marker='.')
    plt.show()

    for cat in range(max(typelist)):
        temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] == cat])
        if np.abs(np.corrcoef(temp, rowvar=False)[0, 1]) < threshold:
            typelist = list(map(lambda x: -1 if x == cat else x, typelist))

    temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] != -1])
    temp_list = np.array([typelist[i] for i in range(len(typelist)) if typelist[i] != -1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(temp[:, 0], temp[:, 1], c=temp_list, marker='.')
    plt.show()

    # clusters = set(temp_list)
    #
    # for k in clusters:
    #     temp = np.array([dataset[i, :] for i in range(len(dataset)) if typelist[i] == k])
    #     temp_list = np.array([typelist[i] for i in range(len(typelist)) if typelist[i] == k])
    #     plt.scatter(temp[:, 0], temp[:, 1], c=temp_list, marker='.')
    #     plt.show()
