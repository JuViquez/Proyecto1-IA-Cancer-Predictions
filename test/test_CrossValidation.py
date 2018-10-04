import pytest

def test_partition_output_0():
    X = [
            [1,2,3],
            [5,6,7],
            [9,10,11],
            [1,2,3],
            [5,6,7],
            [9,10,11],
            [1,2,3],
            [5,6,7],
            [9,10,11]
        ]
    
    y = [4, 8, 12, 4, 8, 12, 4, 8, 12]
    
    k = 2
    fold = 4
    
    expected_X_train = [
                            [1,2,3],
                            [5,6,7],
                            [9,10,11],
                            [1,2,3],
                            [5,6,7],
                            [9,10,11],
                            [1,2,3],
                            [5,6,7]
                        ]
    expected_y_train = [4, 8, 12, 4, 8, 12, 4, 8]
    
    expected_X_test = [[9,10,11]]
    expected_y_test = [8]
    
    X_train, y_train, X_test, y_test  = CrossValidation.partition(X, y, fold, k)
    assert X_train == expected_X_train and  y_train == expected_y_train
       and X_test == expected_X_test and y_test == expected_y_train


def test_partition_output_1():
    X = [
            [1,2,3],
            [5,6,7],
            [9,10,11],
            [1,2,3],
            [5,6,7],
            [9,10,11],
            [1,2,3],
            [5,6,7],
            [9,10,11]
        ]
    
    y = [4, 8, 12, 4, 8, 12, 4, 8, 12]
    
    k = 2
    fold = 1
    
    expected_X_train = [
                            [9,10,11],
                            [1,2,3],
                            [5,6,7],
                            [9,10,11],
                            [1,2,3],
                            [5,6,7],
                            [9,10,11]
                        ]
    expected_y_train = [12, 4, 8, 12, 4, 8, 12]
    
    expected_X_test = [[1,2,3],
                       [5,6,7]]
    expected_y_test = [4,8]
    
    X_train, y_train, X_test, y_test  = CrossValidation.partition(X, y, fold, k)
    assert X_train == expected_X_train and  y_train == expected_y_train
       and X_test == expected_X_test and y_test == expected_y_train


    
    