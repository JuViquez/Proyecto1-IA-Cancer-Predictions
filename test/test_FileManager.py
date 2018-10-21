import pytest
import os
import pandas as pd
import numpy as np
from source.datahandlers.FileManager import csv_to_dataset
from source.datahandlers.FileManager import dataset_to_csv
from source.datahandlers.FileManager import write_to_log


def test_csv_to_dataset_output():
    dataset = pd.read_csv("datasets/test_dataset0.csv")
    assert dataset.equals(csv_to_dataset("datasets/test_dataset0.csv"))


def test_dataset_to_csv_file_isfile():
    if os.path.isfile("datasets/test_dataset.csv"):
        os.remove("datasets/test_dataset.csv")
    dataset = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]
    ]
    dataset = pd.DataFrame(dataset)
    dataset_to_csv("datasets/test_dataset.csv", dataset)
    assert os.path.isfile("datasets/test_dataset.csv")


def test_dataset_to_csv_output():
    if os.path.isfile("datasets/test_dataset.csv"):
        os.remove("datasets/test_dataset.csv")
    dataset = [
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]
    ]
    dataset = pd.DataFrame(dataset)
    dataset_to_csv("datasets/test_dataset.csv", dataset)
    dataset2 = pd.read_csv("datasets/test_dataset.csv")
    assert np.array_equal(dataset.values, dataset2.values)


def test_dataset_to_csv_type_error():
    with pytest.raises(TypeError):
        dataset_to_csv("", 1)


def test_write_to_log_file_isfile():
    if os.path.isfile("logs/test_log.log"):
        os.remove("logs/test_log.log")
    write_to_log("logs/test_log.log", "Test log msg")
    assert os.path.isfile("logs/test_log.log")


def test_write_to_log_type_error():
    with pytest.raises(TypeError):
        write_to_log("logs/test_log.log", 1)


def test_write_to_log_output():
    if os.path.isfile("logs/test_log.log"):
        os.remove("logs/test_log.log")
    test_log_msg = "Test log msg\n"
    write_to_log("logs/test_log.log", test_log_msg)
    f = open("logs/test_log.log", "r")
    content = f.read()
    f.close()
    assert content == test_log_msg
