import numpy as np
from sklearn import datasets

np.random.seed(0)
n_samples = 150


def gen_5_circles():
    theta = np.linspace(0, 2 * np.pi, n_samples)
    radiuses = [.5, 1, 1.5, 2, 2.5]
    points = []
    labels = []
    for i, radius in enumerate(radiuses):
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        points_r = list(map(list, zip(x, y)))
        points += points_r
        labels += [i] * len(points_r)
    return points, labels


def get_blobs():
    random_state = 170
    X, original_labels = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    x = np.dot(X, transformation)
    return x, original_labels


def gen_cos_sin():
    x = [[i, np.cos(i)] for i in np.arange(-5, 5, .3)]
    x = x + [[i, np.sin(i)] for i in np.arange(-5, 5, .3)]
    original_labels = [0] * int(len(x) / 2) + [1] * int(len(x) / 2)
    x = np.array(x)
    original_labels = np.array(original_labels)
    return x, original_labels
