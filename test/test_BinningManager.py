import pytest
from source.datahandlers.BinningManager import BinningManager

@pytest.fixture()
def model():
    ex_bd = BinningManager()
    return ex_bd

def test_binning_data(model):
    X = [
        [2],[3],[1],
        [6],[3.5],[2.5],
        [1.5],[1],[3.5]
    ]
    model.binning_data(X)
    assert model.slices == [[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]]
    assert X[0][0] == "2.0 - 2.5"

def test_convert_data(model):
    X = [
        [2],[3],[1],
        [6],[3.5],[2.5],
        [1.5],[1],[3.5]
    ]
    model.binning_data(X)
    result = [[0],[7]]
    model.convert_data(result,0)
    assert result == [['1.0 - 1.5'],['5.5 - 6.0']]
