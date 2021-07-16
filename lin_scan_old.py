import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import KDTree
from import_coordinates import import_test, import_hs
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.cluster import DBSCAN
from collections import Counter

from sklearn.decomposition import PCA


class visitlist:
    def __init__(self, count):
        self.unvisitedlist = [i for i in range(count)]
        self.visitedlist = list()
        self.unvisitednum = count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1


class SDE:
    def __init__(self, mainpt, a, b, angle):
        self.a = a
        self.b = b
        self.angle = angle
        self.mainpt = mainpt


def dis_two_point(pi, pj):
    dis = ((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2) ** 0.5
    return dis


def calculate_sde(ptindex_array, eps):
    pt_array = dataset[ptindex_array]
    center_pt = pt_array.mean(axis=0)
    pt_trans = pt_array - center_pt
    tempA = np.square(pt_trans)
    A = sum(tempA[:, 0]) - sum(tempA[:, 1])
    C = 2 * sum(pt_trans[:, 0] * pt_trans[:, 1])
    B = (A ** 2 + C ** 2) ** 0.5

    if (C == 0):
        if (-A + B == 0):
            angle = 0
        else:
            angle = math.pi / 2
    else:
        angle = math.atan((-A + B) / C)

    a_ori = math.sqrt(
        sum(np.square(pt_trans[:, 1] * math.sin(angle) + pt_trans[:, 0] * math.cos(angle))) / len(pt_array))
    b_ori = math.sqrt(
        sum(np.square(pt_trans[:, 1] * math.cos(angle) - pt_trans[:, 0] * math.sin(angle))) / len(pt_array))

    if a_ori * b_ori == 0:
        if a_ori > b_ori:
            a_final = float('inf')
            b_final = 0
        else:
            b_final = float('inf')
            a_final = 0
    else:
        trans_indicator = math.sqrt(eps ** 2 / (a_ori * b_ori))
        if a_ori > b_ori:
            a_final = a_ori * trans_indicator
            b_final = b_ori * trans_indicator
        else:
            b_final = a_ori * trans_indicator
            a_final = b_ori * trans_indicator

    result = SDE(center_pt, a_final, b_final, angle)
    return result


def point_in_SDE(pt, SDE):
    [xj, yj] = pt
    [xi, yi] = SDE.mainpt
    if SDE.b == 0:
        if (yj - yi) * math.cos(SDE.angle) - (xj - xi) * math.sin(SDE.angle) == 0:
            return 'in'
        else:
            return 'out'
    elif dis_two_point(pt, SDE.mainpt) > SDE.a:
        return 'out'
    elif dis_two_point(pt, SDE.mainpt) <= SDE.b:
        return 'in'
    else:
        temp = (((yj - yi) * math.sin(SDE.angle) + (xj - xi) * math.cos(SDE.angle)) ** 2 / SDE.a ** 2) + (
                ((yj - yi) * math.cos(SDE.angle) - (xj - xi) * math.sin(SDE.angle)) ** 2 / SDE.b ** 2)
        if temp <= 1:
            return 'in'
        else:
            return 'out'


def union(lst1, lst2):
    if type(lst1) is not list and type(lst1) is not np.ndarray:
        lst1 = [lst1]

    if type(lst2) is not list and type(lst2) is not np.ndarray:
        lst2 = [lst2]

    final_list = list(set(lst1) | set(lst2))
    return final_list


def pts_in_SDE(pts_index, SDE):
    in_list = []
    for ptindex in pts_index:
        pt = dataset[ptindex]
        tag = point_in_SDE(pt, SDE)
        if tag == 'in':
            in_list.append(ptindex)
    return in_list


def lin_scan(dataset, eps, min_pts, threshold, ecc_pts):
    nPoints = dataset.shape[0]
    vPoints = visitlist(nPoints)
    k = -1
    C = [-1 for i in range(nPoints)]
    kd = KDTree(dataset)

    ecc = lambda x: np.sqrt(1 - (x.b ** 2) / (x.a ** 2))

    eccentricities = [ecc(calculate_sde(kd.query(dataset[i], k=ecc_pts)[1], eps))
                      for i in vPoints.unvisitedlist]

    while vPoints.unvisitednum > 0:
        print(vPoints.unvisitednum)
        p = np.argmax(eccentricities)
        vPoints.visit(p)
        N_index = kd.query(dataset[p], k=min_pts)[1]
        sden = calculate_sde(N_index, eps)
        may_in_sden_ptlist = kd.query_ball_point(dataset[p], sden.a)
        pts_in_sden = pts_in_SDE(may_in_sden_ptlist, sden)

        if len(pts_in_sden) >= min_pts:
            eccentricities[p] = 0
            k += 1
            C[p] = k
            for p1 in pts_in_sden:
                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    eccentricities[p1] = 0

                    cluster = [i for i in range(nPoints) if C[i] == k]

                    if len(cluster) < min_pts:
                        M_index = union(cluster, kd.query(dataset[p], k=min_pts)[1])
                    else:
                        M_index = cluster

                    inv_cov = np.linalg.inv(np.cov(dataset[M_index], rowvar=False))

                    # sdem = calculate_sde(M_index, eps)
                    may_in_sdem_ptlist = kd.query_ball_point(dataset[p1], eps * np.sqrt(np.linalg.norm(inv_cov, ord=2)))

                    pts_in_sdem = []
                    mean = dataset[p]
                    for i in may_in_sdem_ptlist:
                        normalized = dataset[i] - mean
                        if np.sqrt(np.transpose(normalized) @ inv_cov @ np.transpose(normalized)) < eps:
                            pts_in_sdem.append(i)

                    _, pca = np.linalg.eigh(inv_cov)

                    pca = pca[-1]
                    pca = pca / np.linalg.norm(pca)

                    if len(pts_in_sdem) >= min_pts:
                        for i in pts_in_sdem:
                            L = np.cov(dataset[kd.query(dataset[i], k=min_pts)[1]], rowvar=False)
                            L_inv = np.linalg.inv(L)
                            _, temp_pca = np.linalg.eigh(L_inv)
                            temp_pca = temp_pca[0]
                            temp_pca = temp_pca / np.linalg.norm(temp_pca)
                            if i not in pts_in_sden and \
                                    np.abs(np.dot(temp_pca, pca)) >= threshold and \
                                    np.sqrt(np.transpose(pca) @ L @ pca) / np.linalg.norm(L, ord=2) > threshold:
                                pts_in_sden.append(i)
                    if C[p1] == -1:
                        C[p1] = k
            if C.count(k) == 1:
                C[p] = -1
            elif vPoints.unvisitednum < 8 / 4 * dataset.shape[0]:
                plt.plot(dataset[p, 0], dataset[p, 1], marker='+')
                plt.scatter(dataset[:, 0], dataset[:, 1], c=C, marker='.')
                plt.show()
                input("Press Enter to continue")

        else:
            eccentricities[p] = 0
            C[p] = -1
    return C


def generate_data(path):
    fw = open(path, 'r')
    pt_array = []
    for line in fw:
        pt = [float(line.split(',')[3]), float(line.split(',')[4])]
        pt_array.append(pt)

    pt_array = np.array(pt_array)
    # plt.figure(figsize=(12, 9), dpi=80)
    # plt.scatter(pt_array[:, 0], pt_array[:, 1], marker='.')
    # plt.show()
    return pt_array


def swiss_roll():
    points, coords = datasets.make_swiss_roll(n_samples=1000, noise=1, random_state=None)
    points = points[:, [0, 2]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], c=coords)
    plt.show()

    return points


def crossing_lines():
    samples = 500

    var = .02
    noises = np.random.normal(0, var, size=[4, samples])

    line_1 = [[t / samples + noises[0, t], t / samples + noises[1, t]] for t in range(samples)]
    line_2 = [[t / samples + noises[2, t] + .5 + 2 * var, 1 - t / samples + noises[3, t] + .55 + 2 * var] for t in
              range(samples)]
    return np.array(line_1 + line_2)


# if __name__ == '__main__':
#     # read data
#     dataset = np.array(import_test())
#
#     # Test: eps = 2, minpts = 10 or 12
#
#     # epsilons = [4, 6]
#     #
#     # minpts_list = [6, 10, 14]
#     #
#     # thresholds = [.1, .2]
#     # [(4, 6, 0.1), (4, 10, 0.1), (4, 14, 0.2), (6, 6, 0.2),
#     # (6, 10, 0.1), (6, 10, 0.2), (6, 10, 0.4), (6, 14, 0.1), (6, 14, 0.2)]
#
#     epsilons = [6]
#
#     minpts_list = [50]
#
#     thresholds = [.95]
#
#     # ADCN method with dataset, eps and minpts
#     for eps in epsilons:
#         for minpts in minpts_list:
#             for threshold in thresholds:
#                 typelist = lin_scan(dataset, eps, minpts, threshold, ecc_pts)
#
#                 plt.scatter(dataset[:, 0], dataset[:, 1], c=typelist, marker='.')
#                 plt.show()

if __name__ == '__main__':
    # read data
    dataset = np.array(import_test())

    dbscan = DBSCAN(eps=5)
    typelist = dbscan.fit_predict(dataset)

    # plt.scatter(dataset[:, 0], dataset[:, 1], c=typelist, marker='.')
    # plt.show()

    clusters = []

    for i in [-1] + list(range(max(typelist))):
        in_cluster = [pt for pt in range(len(dataset)) if typelist[pt] == i]
        clusters.append(in_cluster)

    eps = 7
    minpts = 5
    threshold = .7
    ecc_pts = 10

    categories = []
    sorted_dataset = None

    for cluster in clusters:
        clust_dataset = dataset[cluster]
        if sorted_dataset is None:
            sorted_dataset = clust_dataset
        else:
            sorted_dataset = np.append(sorted_dataset, clust_dataset, axis=0)
        if len(categories) == 0:
            categories = [-1] * len(clust_dataset)
        else:
            typelist = lin_scan(clust_dataset, eps, minpts, threshold, ecc_pts)
            start = max(categories) + 1
            for cat in typelist:
                if cat == -1:
                    categories.append(cat)
                else:
                    categories.append(cat + start)

            plt.scatter(clust_dataset[:, 0], clust_dataset[:, 1], c=typelist, marker='.')
            plt.show()
            # print("hi")

    plt.scatter(sorted_dataset[:, 0], sorted_dataset[:, 1], c=categories, marker='.')
    plt.show()
