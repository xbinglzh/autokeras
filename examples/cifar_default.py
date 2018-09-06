from keras.datasets import cifar10
from autokeras.constant import Constant
from autokeras.generator import CnnGenerator
from autokeras.loss_function import classification_loss
from autokeras.metric import Accuracy
from autokeras.net_transformer import default_transform
from autokeras.preprocessor import DataTransformer, OneHotEncoder
from autokeras.search import train

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    graphs = default_transform(CnnGenerator(10, (32, 32, 3)).generate())

    Constant.MAX_NO_IMPROVEMENT_NUM = 200

    data_transformer = DataTransformer(x_train, augment=True)
    data_transformer.mean = [0.49139968, 0.48215827, 0.44653124]
    data_transformer.std = [0.24703233, 0.24348505, 0.26158768]

    y_encoder = OneHotEncoder()
    y_encoder.fit(y_train)
    y_train = y_encoder.transform(y_train)
    y_test = y_encoder.transform(y_test)

    train_data = data_transformer.transform_train(x_train, y_train)
    test_data = data_transformer.transform_test(x_test, y_test)

    _, _1, graph = train((graphs[0], train_data, test_data, {}, None, Accuracy, classification_loss, True))
