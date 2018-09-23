import os

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
        times.append(item['time'])

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

    _, ax = plt.subplots()
    # red dashes, blue squares and green triangles
    colors = 'rgb'
    label = ['BFS', 'BO', 'AK']
    for i in range(len(paths)):
        ax.plot(indices[i], metric_values[i], colors[i] + ':', label=label[i])

    ax.legend(loc='upper right', fontsize='x-large')
    ax.set_xlabel('Number of Models', fontsize='x-large')
    ax.set_ylabel('Error Rate', fontsize='x-large')
    plt.show()

    _, ax = plt.subplots()
    for i in range(len(paths)):
        ax.plot(times[i], metric_values[i], colors[i] + '--', label=label[i])
    ax.legend(loc='upper right', fontsize='x-large')
    ax.set_xlabel('Seconds', fontsize='x-large')
    ax.set_ylabel('Error Rate', fontsize='x-large')
    plt.show()


if __name__ == '__main__':
    main(['bfs_searcher', 'bo_searcher', 'searcher'])
