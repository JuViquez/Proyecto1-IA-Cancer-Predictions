import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from source.exceptions.EmptyDataset import EmptyDataset
from source.datahandlers.DataManager import DataManager


def test_preprocess_data_empty_array_exception():
    data_manager = DataManager([])
    with pytest.raises(EmptyDataset):
        data_manager.preprocess_data(0)


def test_preprocess_data_output():
    dataset = [
        [10000.0, 50.0, 10.0, 100.0],
        [5000.0, 100.0, 20.0, 35.0],
        [1000.0, 10.0, 0.0, 12.0]
    ]
    data_manager = DataManager(dataset)
    dataset = pd.DataFrame(dataset)
    sc = StandardScaler()
    output_X = sc.fit_transform(dataset.iloc[:, :-1].values)
    X, y = data_manager.preprocess_data(-1)
    assert np.array_equal(X, output_X) and np.array_equal(
        y, dataset.iloc[:, 3].values)


def test_split_X_y_output():
    dataset = [
        [10000, 50, 10, 100],
        [5000, 100, 20, 35],
        [1000, 10, 0, 12]
    ]
    data_manager = DataManager(dataset)
    dataset = pd.DataFrame(dataset)
    X, y = data_manager.split_X_y(-1)
    assert np.array_equal(
        X, dataset.iloc[:, :-1].values) and np.array_equal(y, dataset.iloc[:, 3].values)


def test_add_column_result():
    dataset = [
        [10000, 50],
        [5000, 100],
        [1000, 10]
    ]
    column = [1, 2, 3]

    expected_result = pd.DataFrame([
        [10000, 50, 1],
        [5000, 100, 2],
        [1000, 10, 3]
    ]
    )
    data_manager = DataManager(dataset)
    data_manager.add_column(column, "Prediction")
    assert np.array_equal(expected_result.values, data_manager.dataset.values)


def test_shuffle_dataset_result():
    dataset = [
        [10000, 50, 10, 100],
        [5000, 100, 20, 35],
        [1000, 10, 0, 12],
        [10000, 50, 10, 100],
        [5000, 100, 20, 35],
        [1000, 10, 0, 12],
        [1000, 10],
        [4234, 12],
        [12113, 9]
    ]

    data_manager = DataManager(dataset)
    data_manager.shuffle_dataset()
    assert not np.array_equal(data_manager.dataset, pd.DataFrame(dataset))


def test_create_encoder_output():
    labels = ['a', 'b', 'c', 'c']
    expected_label_encoder = LabelEncoder()
    expected_output = expected_label_encoder.fit(labels)

    data_manager = DataManager([])
    actual_output = data_manager.create_encoder(labels)
    assert np.array_equal(
        expected_output.transform(labels),
        actual_output.transform(labels))


def test_split_train_test_output():
    X = [
        [10000, 50],
        [5000, 100],
        [1000, 10],
        [4234, 12],
        [12113, 9]
    ]
    y = [2, 3, 4, 5, 6]

    data_manager = DataManager([])

    X_train, y_train, X_test, y_test = data_manager.split_train_test(
        X, y, test_size=0.25)

    expected_X_train = X[1:]
    expected_y_train = y[1:]

    expected_X_test = [X[0]]
    expected_y_test = [y[0]]

    assert np.array_equal(expected_X_train, X_train)
    assert np.array_equal(expected_y_train, y_train)

    assert np.array_equal(expected_X_test, X_test)
    assert np.array_equal(expected_y_test, y_test)


def test_split_train_trest_value_error():
    test_size_lt_zero = -1
    test_size_gt_one = 2
    data_manager = DataManager([])
    with pytest.raises(ValueError):
        data_manager.split_train_test([], [], test_size_lt_zero)
    with pytest.raises(ValueError):
        data_manager.split_train_test([], [], test_size_gt_one)
