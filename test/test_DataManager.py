import pytest
from sklearn.preprocessing import StandardScaler
import pandas as pd

def test_preprocess_data_empty_array_exception():
    data_manager = DataManager([[]])
    with pytest.raises(EmptyArray):
           data_manager.preprocess_data([])

def test_preprocess_data_column_data_type_exception():
    dataset = [
                [10000, 'b', 10],
                [5000, 'c', 0],
                [1000, 0, 12]
              ]
    data_manager = DataManager(dataset)
    with pytest.raises(ColumnDataType):
           data_manager.preprocess_data([-1])

def test_preprocess_data_output():
    dataset = [
                [10000, 50, 10, 100],
                [5000, 100, 20, 35],
                [1000, 10, 0, 12]
              ]
    dataset = pd.DataFrame(dataset)
    data_manager = DataManager(dataset)
    sc = StandardScaler()
    output_X = sc.fit_transform(dataset.iloc[:, :-1].values)
    X, y = data_manager.preprocess_data(-1) 
    assert X == output_X and y == dataset.iloc[:, 3].values


def test_preprocess_data_output_dataset_with_nulls():
    dataset = [
                [100, 100, 10],
                [50,  50],
                [None, 10]
              ]
    dataset = pd.DataFrame(dataset)
    data_manager = DataManager(dataset)
    
    dataset.loc[2,0] = 75
    dataset.loc[1,2] = 10
    dataset.loc[2,2] = 10
    y_test = dataset[1].values
    sc = StandardScaler()
    
    dataset = dataset.drop(columns=[1])
    output_X = sc.fit_transform(dataset.values)
  
    X, y = data_manager.preprocess_data([1]) 
    assert X == output_X and y == y_test

    
