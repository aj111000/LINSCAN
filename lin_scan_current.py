import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from scipy.linalg import sqrtm as sqrtm
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


def kl_dist(x, y):
    # (x_1, x_2, s_11, s_12, s_22)
    cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
    inv1 = np.linalg.inv(cov1)
    p1 = np.array([x[0], x[1]])

    cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
    inv2 = np.linalg.inv(cov2)
    p2 = np.array([y[0], y[1]])

    return (p1 - p2).transpose() @ (inv1 + inv2) @ (p1 - p2) + np.trace(inv1 @ cov2 + inv2 @ cov1) - 4


def fi_dist(x, y):
    # (x_1, x_2, s_11, s_12, s_22)
    cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
    p1 = np.array([x[0], x[1]])

    cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
    p2 = np.array([y[0], y[1]])

    return np.linalg.norm(p1 - p2) ** 2 + np.trace(cov1 + cov2 - 2 * sqrtm(cov1 @ cov2))


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


def fi_db_scan(dataset, eps, min_pts):
    nPoints = dataset.shape[0]
    vPoints = VisitList(nPoints)
    current_cat = -1
    categories = [-1 for i in range(nPoints)]
    kd = BallTree(dataset, metric="pyfunc", func=fi_dist)

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


def fi_embed_scan(dataset, eps, min_pts, ecc_pts):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

        embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
    embeddings = np.array(embeddings)

    return fi_db_scan(embeddings, eps, min_pts)


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


if __name__ == '__main__':
    from lin_scan_old import swiss_roll, crossing_lines

    # read data
    dataset = np.array(import_hs())

    dataset /= np.max(np.abs(dataset))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(dataset[:, 0], dataset[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.show()

    x_range = [-1, 1]
    y_range = [-1, 0]

    x_filt = lambda x: x_range[0] <= x <= x_range[1]
    y_filt = lambda y: y_range[0] <= y <= y_range[1]

    filt = lambda pt: x_filt(pt[0]) and y_filt(pt[1])

    dataset = np.array(list(filter(filt, dataset.tolist())))
    dataset /= np.max(np.abs(dataset), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal',adjustable='box')

    plt.scatter(dataset[:, 0], dataset[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.show()

    eps = 8
    min_pts = 10
    threshold = .5
    ecc_pts = 10

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

