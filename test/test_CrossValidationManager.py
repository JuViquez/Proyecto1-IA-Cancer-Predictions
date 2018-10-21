import pytest
import numpy as np
import tensorflow as tf
from source.cross_validation.CrossValidationManager import CrossValidationManager
from source.models.neural_network.NeuralNetwork import NeuralNetwork
from source.utilities.Metrics import l1_loss


@pytest.fixture
def neural_network():
    return NeuralNetwork(
        5,
        5,
        3,
        tf.nn.relu,
        tf.nn.softmax,
        'binary_crossentropy',
        'sgd',
        10)


def test_partition_output_0():
    X = [
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11]
    ]

    y = [4, 8, 12, 4, 8, 12, 4, 8, 12]

    k = 2
    fold = 1

    expected_X_train = [
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3]
    ]
    expected_y_train = [4, 8, 12, 4]

    expected_X_test = [
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3],
        [5, 6, 7]
    ]
    expected_y_test = [8, 12, 4, 8]

    test_nw = neural_network()
    cvm = CrossValidationManager(test_nw, X, y, l1_loss, k)

    X_train, y_train, X_test, y_test = cvm.partition(fold)
    assert X_train == expected_X_train
    assert y_train == expected_y_train
    assert X_test == expected_X_test
    assert y_test == expected_y_test


def test_partition_value_error():
    cvm = CrossValidationManager(None, None, None, None, 2)
    with pytest.raises(ValueError):
        cvm.partition(2)


def test_error_rate_output():
    test_nw = neural_network()
    X = np.array([
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11],
        [1, 2, 3],
        [5, 6, 7],
        [9, 10, 11]
    ])
    y = np.array([1, 1, 2, 2, 2, 2, 1, 1, 2])
    test_nw.fit(X, y)

    num_samples = len(X)
    expected_error_rate = 0
    for i in range(num_samples):
        hx = test_nw.predict(X[i])
        expected_error_rate += l1_loss(y[i], hx)
    expected_error_rate = expected_error_rate / num_samples

    cvm = CrossValidationManager(test_nw, X, y, l1_loss)
    actual_error_rate = cvm.error_rate(X, y)
    assert expected_error_rate == actual_error_rate
