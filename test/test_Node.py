import pytest
from source.models.decisiontree.Node import Node


@pytest.fixture()
def model():
    example_node = Node()
    return example_node


def test_count_rows_2d_array(model):
    dataset = [
        ['1'], ['0'],
        ['1'], ['1'],
        ['0'], ['1'],
        ['0'], ['1']
    ]
    result = {}
    result['0'] = 3
    result['1'] = 5
    assert model.count_rows(dataset, 0) == result


def test_count_rows_list(model):
    dataset = [
        'Yes', 'No', 'No',
        'No', 'Yes', 'No',
        'Yes', 'No', 'Yes'
    ]
    result = {}
    result['Yes'] = 4
    result['No'] = 5
    assert model.count_rows(dataset, 0) == result


def test_count_probability(model):
    dataset = [
        ['1'], ['0'],
        ['1'], ['1'],
        ['0'], ['1'],
        ['0'], ['1']
    ]
    classification = [
        'Yes', 'Yes', 'No', 'Yes',
        'Yes', 'No', 'No', 'No'
    ]
    assert model.count_probability(dataset, classification, 0, '1') == 2


def test_entropy_50(model):
    assert model.entropy(0.5) == 1


def test_entropy_100(model):
    assert model.entropy(1) == 0


def test_remainder(model):
    dataset = [
        ['1'], ['1'], ['1'],
        ['0'], ['0'], ['0'],
        ['1'], ['1'], ['0'],
        ['1'], ['0'], ['0'],
    ]
    classification = [
        'Yes', 'No', 'No',
        'No', 'Yes', 'No',
        'Yes', 'No', 'Yes',
        'Yes', 'No', 'Yes'
    ]
    assert model.remainder(dataset, classification, 0) == 1


def test_split_features(model):
    features = model.split_features(5)
    assert len(features) == 2


def test_best_gain_value(model):
    model.gain = 0
    model.column = 0
    dataset = [
        ['0'], ['0'], ['0'],
        ['0'], ['0'], ['0'],
        ['1'], ['1'], ['1'],
        ['1'], ['1'], ['1'],
    ]
    classification = [
        'Yes', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes',
        'No', 'No', 'No',
        'No', 'No', 'No'
    ]
    model.best_gain(dataset, classification)
    assert round(model.gain, 2) == 1


def test_best_gain_column(model):
    model.gain = 0
    model.column = 0
    dataset = [
        ['T', '0'], ['F', '0'], ['T', '0'],
        ['F', '0'], ['T', '0'], ['F', '0'],
        ['T', '1'], ['F', '1'], ['T', '1'],
        ['F', '1'], ['T', '1'], ['F', '1'],
    ]
    classification = [
        'Yes', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes',
        'No', 'No', 'No',
        'No', 'No', 'No'
    ]
    model.best_gain(dataset, classification)
    assert model.column == 0 or model.column == 1


def test_plurality_empty(model):
    assert not model.plurality({})


def test_plurality(model):
    prediction = {"Yes": 5, "No": 7}
    assert model.plurality(prediction) == {"No": 7}
