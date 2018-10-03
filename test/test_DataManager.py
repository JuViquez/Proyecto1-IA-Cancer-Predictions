import pytest
from sklearn.preprocessing import StandardScaler
import pandas as pd

def test_preprocess_data_empty_array_exception():
    data_manager = DataManager([])
    with pytest.raises(EmptyDataset):
           data_manager.preprocess_data([])

def test_preprocess_data_output():
    dataset = [
                [10000, 50, 10, 100],
                [5000, 100, 20, 35],
                [1000, 10, 0, 12]
              ]
    data_manager = DataManager(dataset)
    dataset = pd.DataFrame(dataset)
    sc = StandardScaler()
    output_X = sc.fit_transform(dataset.iloc[:, :-1].values)
    X, y = data_manager.preprocess_data(-1) 
    assert X == output_X and y == dataset.iloc[:, 3].values

def test_split_X_y_output():
    dataset = [
                [10000, 50, 10, 100],
                [5000, 100, 20, 35],
                [1000, 10, 0, 12]
              ]
    data_manager = DataManager(dataset)
    dataset = pd.DataFrame(dataset)
    X, y = data_manager.split_X_y(-1)
    assert X == dataset.iloc[:, :-1].values and y == dataset.iloc[:, 3].values
    
def test_add_column_result():
    dataset = [
                [10000, 50],
                [5000, 100],
                [1000, 10]
              ]
    column = [1,2,3]
    
    expected_result = pd.DataFrame(  [
                [10000, 50, 1],
                [5000, 100, 2],
                [1000, 10, 3]
              ]
            )
    data_manager = DataManager(dataset)
    data_manager.add_column(column)
    assert expected_result.equals(data_manager.dataset)
    
def test_shuffle_dataset_result():
    dataset = [
                [10000, 50, 10, 100],
                [5000, 100, 20, 35],
                [1000, 10, 0, 12]
              ]
    data_manager = DataManager(dataset)
    data_manager.shuffle_dataset()
    assert not data_manager.dataset.equals(pd.DataFrame(dataset)) 
    
    
    
    
    
