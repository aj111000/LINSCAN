import numpy as np
import numpy.random as rand


def gen_data(lin_clusts=10,
             lin_num=100,
             lin_R=2,
             iso_clusts=5,
             iso_num=200,
             iso_R=10,
             noise_num=120,
             x_min=-50,
             x_max=50,
             y_min=-50,
             y_max=50):
    def gen(clusts, num_pts, R, len_mult=1):
        data = []
        lab = []

        unif = lambda: rand.uniform(low=-.5, high=.5)

        for i in range(clusts):
            x = unif() * (x_max - x_min)
            y = unif() * (y_max - y_min)
            s = unif() * np.pi
            length = (rand.uniform() + .2) * 10 * len_mult + 1
            top_x = x + length * np.sin(s)
            bot_x = x - length * np.sin(s)
            top_y = y + length * np.cos(s)
            bot_y = y - length * np.cos(s)

            dx = (bot_x - top_x) / (num_pts - 1)
            dy = (bot_y - top_y) / (num_pts - 1)
            for j in range(num_pts):
                x1 = top_x + dx * j
                y1 = top_y + dy * j

                ddx = unif() * R * (rand.uniform() + .1)
                ddy = unif() * R * (rand.uniform() + .1)
                data.append([x1 + ddx, y1 + ddy])
                lab.append(i)

        return data, lab

    data = []
    labels = []

    lin_data, lin_labels = gen(lin_clusts, lin_num, lin_R)

    data = [*data, *lin_data]
    labels = [*labels, *lin_labels]

    iso_data, iso_labels = gen(iso_clusts, iso_num, iso_R, len_mult=0)

    iso_labels = list(map(lambda x: -1, iso_labels))

    data = [*data, *iso_data]
    labels = [*labels, *iso_labels]

    noise_data = []
    noise_labels = []

    for j in range(noise_num):
        noise_data.append(
            [rand.uniform(x_min, x_max),
             rand.uniform(y_min, y_max)]
        )
        noise_labels.append(-1)

    data = [*data, *noise_data]
    labels = [*labels, *noise_labels]

    return data, labels


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from clust_scoring import cluster_accuracy

    data, labels = gen_data()
    data = np.array(data)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', s=(2 * 72. / fig1.dpi) ** 2)
    plt.show()
