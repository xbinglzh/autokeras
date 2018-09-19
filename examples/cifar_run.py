import pickle
import time
from functools import reduce

import torch
import numpy as np

import os
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from autokeras.constant import Constant
from autokeras.image_supervised import run_searcher_once
from autokeras.loss_function import classification_loss
from autokeras.metric import Accuracy
from autokeras.preprocessor import DataTransformer, OneHotEncoder
from autokeras.search import train, Searcher
from autokeras.utils import temp_folder_generator, pickle_from_file, pickle_to_file, ensure_dir


def load_searcher(path):
    return pickle_from_file(os.path.join(path, 'searcher'))


def save_searcher(path, searcher):
    pickle.dump(searcher, open(os.path.join(path, 'searcher'), 'wb'))


def main(searcher, path):
    (x_final, y_final), (x_eval, y_eval) = cifar10.load_data()

    # graphs = default_transform(CnnGenerator(10, (32, 32, 3)).generate())

    Constant.MAX_NO_IMPROVEMENT_NUM = 200
    # Constant.SEARCH_MAX_ITER = 5
    # Constant.MAX_ITER_NUM = 5

    time_limit = 12 * 60 * 60

    start_time = time.time()
    time_remain = time_limit
    ensure_dir(path)
    print(path)

    input_shape = x_final.shape[1:]
    searcher_args = {'n_output_node': 10, 'input_shape': input_shape, 'path': path,
                     'metric': Accuracy, 'loss': classification_loss, 'verbose': True}
    searcher = searcher(**searcher_args)
    save_searcher(path, searcher)

    x_train, x_test, y_train, y_test = train_test_split(x_final, y_final,
                                                        test_size=min(Constant.VALIDATION_SET_SIZE,
                                                                      int(len(y_final) * 0.2)),
                                                        random_state=42)

    data_transformer = DataTransformer(x_train, augment=True)
    data_transformer.mean = [0.49139968, 0.48215827, 0.44653124]
    data_transformer.std = [0.24703233, 0.24348505, 0.26158768]

    y_encoder = OneHotEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    y_test = y_encoder.transform(y_test)
    y_eval = y_encoder.transform(y_eval)
    y_final = y_encoder.transform(y_final)
    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)
    final_data = data_transformer.transform_train(x_final, y_final)
    eval_data = data_transformer.transform_test(x_eval, y_eval)

    try:
        while time_remain > 0:
            run_searcher_once(train_data, test_data, path, int(time_remain))
            if len(load_searcher(path).history) >= Constant.MAX_MODEL_NUM:
                break
            time_elapsed = time.time() - start_time
            time_remain = time_limit - time_elapsed
    except TimeoutError:
        if len(load_searcher(path).history) == 0:
            raise TimeoutError("Search Time too short. No model was found during the search time.")
        print('Time is out.')

    # final train
    searcher = load_searcher(path)
    graph = searcher.load_best_model()

    metric_value, loss, graph = train((graph, final_data, eval_data, {}, None, Accuracy, classification_loss, True))
    pickle_to_file(graph, os.path.join(path, 'best_model'))

    model = graph.produce_model()
    model.eval()

    outputs = []
    with torch.no_grad():
        for index, (inputs, _) in enumerate(eval_data):
            outputs.append(model(inputs).numpy())
    output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
    # predicted = y_encoder.inverse_transform()
    print(Accuracy.compute(output, y_eval))


if __name__ == '__main__':
    main(Searcher, '/home/haifeng/ak/searcher-cifar10')
