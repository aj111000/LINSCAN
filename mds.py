import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from scipy.linalg import sqrtm as sqrtm
import sklearn.neighbors as neighbors
from scipy.spatial.distance import jensenshannon
from lin_scan_old_2 import kl_dist

from import_coordinates import import_test, import_hs

from sklearn.manifold import MDS

dataset = np.array(import_test())

kd = KDTree(dataset)

ecc_pts = 20

embeddings = []
for p in range(len(dataset)):
    cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
    cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
    # cov /= max(np.linalg.eig(cov)[0])
    mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)

    embeddings.append(np.concatenate([mean, [cov[0, 0], cov[0, 1], cov[1, 1]]]))
dataset = np.array(embeddings)

embedding = MDS(n_components=3, metric=False, dissimilarity='precomputed')

D = np.zeros([dataset.shape[0], dataset.shape[0]])

for i in range(dataset.shape[0]):
    for j in range(dataset.shape[0] - i):
        D[i, i + j] = kl_dist(dataset[i, :], dataset[i + j, :])
        D[i + j, i] = D[i, i + j]

embeddings = embedding.fit_transform(D)

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.scatter(embeddings[:, 0], embeddings[:, 1],embeddings[:, 2], marker='o', s=(2 * 72. / fig.dpi) ** 2)
plt.show()
