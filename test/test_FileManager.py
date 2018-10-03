import pytest
import os
import pandas as pd

@pytest.fixture
def dataset_path(filename):
    script_dir = os.path.dirname(__file__)
    return  os.path.join(script_dir+"/../datasets", "/" + filename )

@pytest.fixture
def log_path(filename):
    script_dir = os.path.dirname(__file__)
    return  os.path.join(script_dir+"/../logs", "/" + filename )

def test_csv_to_dataset_output():
    file_manager = FileManager()
    dataset = pd.read_csv(dataset_path("test_dataset.csv"))
    assert dataset.equals(file_manager.csv_to_dataset("test_dataset.csv"))

def test_dataset_to_csv_file_exist():
    if os.path.exist(dataset_path("dataset_to_csv.csv")):
        os.remove(dataset_path("dataset_to_csv.csv"))
    file_manager = FileManager()
    dataset = pd.read_csv(dataset_path("test_dataset.csv"))
    file_manager.dataset_to_csv("dataset_to_csv.csv", dataset)
    assert os.path.exist(dataset_path("dataset_to_csv.csv"))

def test_dataset_to_csv_output():
    if os.path.exist(dataset_path("dataset_to_csv.csv")):
        os.remove(dataset_path("dataset_to_csv.csv"))
    file_manager = FileManager()
    dataset = pd.read_csv(dataset_path("test_dataset.csv"))
    file_manager.dataset_to_csv("dataset_to_csv.csv", dataset)
    dataset2 = pd.read_csv(dataset_path("dataset_to_csv.csv"))
    assert dataset.equals(dataset2)

def test_dataset_to_csv_type_error():
    file_manager = FileManager()
    with pytest.raises(TypeError):
         file_manager.dataset_to_csv("", 1)
         
def test_write_to_log_file_exist():
    if os.path.exist(log_path("test_log.log")):
        os.remove(log_path("test_log.log"))
    file_manager = FileManager()
    file_manager.write_to_log("test_log.log", "Test log msg")
    assert os.path.exist(log_path("test_log.log"))
    
def test_write_to_log_type_error():
    file_manager = FileManager()
    with pytest.raises(TypeError):
         file_manager.write_to_log("test_log.log", 1)
         
def test_write_to_log_output():
    if os.path.exist(log_path("test_log.log")):
        os.remove(log_path("test_log.log"))
    file_manager = FileManager()
    test_log_msg = "Test log msg\n"
    file_manager.write_to_log("test_log.log", test_log_msg)
    f = open(log_path(test_log_msg), "r")
    content = f.read()
    f.close()
    assert content == test_log_msg
         

    


    
    