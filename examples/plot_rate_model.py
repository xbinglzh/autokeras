import os

import matplotlib

from autokeras.utils import pickle_from_file
import numpy as np
import matplotlib.pyplot as plt


def load_searcher(path):
    return pickle_from_file(path)


def get_data(path):
    searcher = load_searcher(path)

    indices = []
    times = []
    metric_values = []

    for index, item in enumerate(searcher.history):
        indices.append(index)
        metric_values.append(1 - item['metric_value'])
        times.append(item['time'] / 60.0 / 60.0)

    for i in range(1, len(times)):
        times[i] = times[i - 1] + times[i]
        metric_values[i] = min(metric_values[i], metric_values[i - 1])

    return indices, times, metric_values


def main(paths):
    indices = []
    times = []
    metric_values = []
    for path in paths:
        a, b, c = get_data(path)
        indices.append(a)
        times.append(b)
        metric_values.append(c)
    # evenly sampled time at 200ms intervals
    # t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    colors = 'rgb'
    label = ['BFS', 'BO', 'AK']

    font = {'size': 18}

    matplotlib.rc('font', **font)
    # matplotlib.rcParams['figure.figsize'] = 2, 5

    _, ax = plt.subplots()
    for i in range(len(paths)):
        ax.plot(indices[i], metric_values[i], colors[i] + '--', label=label[i])
    ax.legend(loc='upper right')
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Error Rate')
    plt.ylim((0.15, 0.25))
    plt.xlim(0, 70)
    plt.show()

    _, ax = plt.subplots()
    for i in range(len(paths)):
        ax.plot(times[i], metric_values[i], colors[i] + '--', label=label[i])
    ax.legend(loc='upper right')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Error Rate')
    plt.ylim((0.15, 0.25))
    plt.xlim(0, 12)
    plt.show()


if __name__ == '__main__':
    main(['bfs_searcher', 'bo_searcher', 'searcher'])
