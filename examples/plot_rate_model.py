import os

from autokeras.utils import pickle_from_file
import numpy as np
import matplotlib.pyplot as plt


def load_searcher(path):
    return pickle_from_file(os.path.join(path, 'searcher'))


def get_data(path):
    searcher = load_searcher(path)

    model_num_rate = []
    time_rate = []

    for index, item in enumerate(searcher.history):
        model_num_rate.append((index, item['metric_value']))
        time_rate.append((item['time'], item['metric_value']))

    return model_num_rate, time_rate


def main():
    n_rates = []
    t_rates = []
    paths = []

    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()


main()
