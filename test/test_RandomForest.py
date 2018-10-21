import pytest
from numpy import array
from source.models.randomforest.RandomForest import RandomForest
from source.models.decisiontree.DecisionTree import DecisionTree


@pytest.fixture()
def model():
    ex_rf = RandomForest(2)
    return ex_rf


@pytest.fixture()
def X_data():
    dataset = [
        ['Yes', 'No', 'No', 'Yes', 'Some', '3', 'No', 'Yes', 'French', '10'],
        ['Yes', 'No', 'No', 'Yes', 'Full', '1', 'No', 'No', 'Thai', '60'],
        ['No', 'Yes', 'No', 'No', 'Some', '1', 'No', 'No', 'Burger', '10'],
        ['Yes', 'No', 'Yes', 'Yes', 'Full', '1', 'Yes', 'No', 'Thai', '30'],
        ['Yes', 'No', 'Yes', 'No', 'Full', '3', 'No', 'Yes', 'French', '80'],
        ['No', 'Yes', 'No', 'Yes', 'Some', '2', 'Yes', 'Yes', 'Italian', '10'],
        ['No', 'Yes', 'No', 'No', 'None', '1', 'Yes', 'No', 'Burger', '10'],
        ['No', 'No', 'No', 'Yes', 'Some', '2', 'Yes', 'Yes', 'Thai', '10'],
        ['No', 'Yes', 'Yes', 'No', 'Full', '1', 'Yes', 'No', 'Burger', '80'],
        ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '3', 'No', 'Yes', 'Italian', '30'],
        ['No', 'No', 'No', 'No', 'None', '1', 'No', 'No', 'Thai', '10'],
        ['Yes', 'Yes', 'Yes', 'Yes', 'Full', '1', 'No', 'No', 'Burger', '60']
    ]
    return array(dataset)


def test_split_dataset(model, X_data):
    Y = array(['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
               'No', 'Yes', 'No', 'No', 'No', 'Yes'])
    dataset_size = round(len(X_data) / model.size)
    X, Y = model.split_dataset(X_data, Y, dataset_size)
    assert len(X) == dataset_size and len(Y) == dataset_size


def test_fit_size_0(model, X_data):
    model.size = 0
    model.fit(X_data, array([]))
    assert len(model.trees) == 0


def test_fit(model, X_data):
    Y = array(['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
               'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'])
    model.fit(X_data, Y)
    assert len(model.trees) == 2
    assert isinstance(model.trees[0], DecisionTree)
    assert isinstance(model.trees[1], DecisionTree)


def test_predict_None(model, X_data):
    model.size = 12
    Y = array(['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
               'No', 'Yes', 'No', 'No', 'No', 'Yes'])
    model.fit(X_data, Y)
    result = model.predict(array(['Maybe',
                                  'Maybe',
                                  'Maybe',
                                  'Maybe',
                                  'Maybe',
                                  '4',
                                  'Maybe',
                                  'Maybe',
                                  'Pizza',
                                  '100']))
    assert result == "?"


def test_predict(model, X_data):
    model.size = 7
    Y = array(['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
               'No', 'Yes', 'No', 'No', 'No', 'Yes'])
    model.fit(X_data, Y)
    result = model.predict(
        array(['Yes', 'Yes', 'Yes', 'Yes', 'Full', '1', 'No', 'No', 'Burger', '60']))
    assert result == "Yes"
