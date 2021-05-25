import numpy as np


def adcn_cluster(points: np.array, eps: float, min_pts: int):
    points = [[idx, point] for idx, point in enumerate(points)]
    visited = [0 for _ in range(len(points))]
    label = [None for _ in range(len(points))]  # 0 is noise, 1 is first cluster 2 is second, etc.

    cluster = 0

    for item in points:
        idx = item[0]
        point = item[1]
        visited[idx] = 1
        neighborhood = get_neighborhood([idx, point], points, eps)

        if len(neighborhood) < min_pts:
            label[idx] = 0
        else:
            cluster += 1
            label[idx] = cluster

            while neighborhood:
                neigh_item = neighborhood.pop(0)
                neigh_idx = neigh_item[0]
                neigh = neigh_item[1]
                if neigh_idx == idx:
                    continue

                if not visited[neigh_idx]:
                    visited[neigh_idx] = 1
                    sub_neighborhood = get_neighborhood([neigh_idx, neigh], points, eps)
                    if len(sub_neighborhood) >= min_pts:
                        neighborhood = neighborhood + sub_neighborhood
                        label[neigh_idx] = cluster
                    else:
                        label[neigh_idx] = 0

    return label


def get_neighborhood(point, points, eps):
    circ_neighborhood = [neigh
                         for neigh in points
                         if 0 < np.linalg.norm(np.array(point[1]) - np.array(neigh[1])) < eps]

    return calc_ellipse(point, circ_neighborhood, points, eps)


def calc_ellipse(center, neighborhood, points, eps):
    if not neighborhood:
        return []

    center_idx = center[0]
    center_loc = center[1]

    centering = lambda x: [x[0], np.array(x[1]) - np.array(center_loc)]

    cent_points = list(map(centering, points))

    cent_idx = [point[0] for point in cent_points]
    cent_points = [point[1] for point in cent_points]

    cent_points = np.array(cent_points)

    x = cent_points[:, 0]
    y = cent_points[:, 1]

    A = np.dot(x, x) - np.dot(y, y)
    C = 2 * np.dot(x, y)
    B = np.sqrt(A ** 2 + C ** 2)

    theta_1 = np.arctan(-(A + B) / C)
    theta_2 = np.arctan(-(A - B) / C)

    sd_1 = calc_sd(theta_1, neighborhood)
    sd_2 = calc_sd(theta_2, neighborhood)

    if sd_1 < sd_2:
        theta = theta_2
        a = sd_2
        b = sd_1
    else:
        theta = theta_1
        a = sd_1
        b = sd_2

    a = eps * np.sqrt(a / b)
    b = eps * np.sqrt(b / a)

    ellipse = []
    for idx, point in zip(cent_idx, cent_points):
        if center_idx == idx:
            continue
        dist_x = ((point[1] * np.sin(theta)
                   + point[0] * np.cos(theta)) ** 2) / (a ** 2)
        dist_y = ((point[1] * np.cos(theta)
                   + point[0] * np.sin(theta)) ** 2) / (b ** 2)

        if dist_x + dist_y <= 1:
            ellipse.append([idx, point])

    return ellipse


def calc_sd(theta, points):
    sum = 0
    for point in points:
        point = point[1]
        sum += (point[0] * np.cos(theta) + point[1] * np.sin(theta)) ** 2

    return np.sqrt(sum / len(points))


if __name__ == '__main__':
    points = np.random.uniform(0, 1, size=[1000, 2])
    eps = 1
    min_pts = 5
    label = adcn_cluster(points, eps, min_pts)

    import matplotlib.pyplot as plt

    plt.scatter(points[:, 0], points[:, 1], c=label)
    plt.show()
